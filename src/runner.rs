// src/runner.rs

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use tokio::{
    fs::File as TokioFile,
    io::{AsyncBufReadExt, BufReader},
    process::{Child, Command},
    sync::{Mutex, RwLock},
    time::sleep,
};
use anyhow::{Result, bail, Context};
use tracing::{info, error, debug, warn};
use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::config::{ConfigManager, ModelConfig, RunnerConfig}; // Assuming config.rs is in the same crate

// Health Status Constants
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum HealthStatus {
    Ok,
    Loading,
    Error,
    NotRunning,
    NotLoaded,
}

impl HealthStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            HealthStatus::Ok => "ok",
            HealthStatus::Loading => "loading",
            HealthStatus::Error => "error",
            HealthStatus::NotRunning => "not_running",
            HealthStatus::NotLoaded => "not_loaded",
        }
    }
}

// Health Messages
pub struct HealthMessages;

impl HealthMessages {
    pub const READY: &'static str = "Ready";
    pub const MODEL_LOADING: &'static str = "Model is still loading";
    pub const RUNNER_NOT_RUNNING: &'static str = "Runner not running";
    pub const MODEL_NOT_LOADED: &'static str = "Model not loaded in runner";
    pub const NO_RUNNER_AVAILABLE: &'static str = "No runner available";
    pub const HEALTH_CHECK_TIMEOUT: &'static str = "Health check timeout";
    pub const CONNECTION_ERROR: &'static str = "Connection error";
    pub const HEALTH_CHECK_FAILED: &'static str = "Health check failed";
}

/// Represents a single llama.cpp server process.
pub struct RunnerProcess {
    pub runner_name: String,
    runner_config: RunnerConfig,
    pub host: String,
    pub port: u16,
    session_log_dir: PathBuf,
    process: Arc<Mutex<Option<Child>>>,
    output_file: Arc<Mutex<Option<TokioFile>>>, // Use TokioFile for async file ops
    models: Arc<RwLock<Vec<ModelConfig>>>, // Models this runner is responsible for
    current_model: Arc<RwLock<Option<ModelConfig>>>, // Model currently loaded by the process
    is_starting: Arc<Mutex<bool>>,
    reqwest_client: Client, // Re-use reqwest client for health checks and forwarding
}

impl RunnerProcess {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runner_name: String,
        runner_config: RunnerConfig,
        host: String,
        port: u16,
        session_log_dir: PathBuf,
    ) -> Self {
        Self {
            runner_name,
            runner_config,
            host,
            port,
            session_log_dir,
            process: Arc::new(Mutex::new(None)),
            output_file: Arc::new(Mutex::new(None)),
            models: Arc::new(RwLock::new(Vec::new())),
            current_model: Arc::new(RwLock::new(None)),
            is_starting: Arc::new(Mutex::new(false)),
            reqwest_client: Client::new(),
        }
    }

    pub async fn add_model(&self, model_config: ModelConfig) {
        let mut models = self.models.write().await;
        models.push(model_config);
    }

    pub async fn get_model_by_alias(&self, model_alias: &str) -> Option<ModelConfig> {
        let models = self.models.read().await;
        models.iter().find(|m| {
            m.model_alias.as_deref().unwrap_or_else(|| m.model.file_name().unwrap().to_string_lossy().as_ref()) == model_alias
        }).cloned()
    }

    pub async fn is_model_loaded(&self, model_alias: &str) -> bool {
        let current_model_lock = self.current_model.read().await;
        match current_model_lock.as_ref() {
            Some(model_config) => {
                let current_alias = model_config.model_alias.as_deref().unwrap_or_else(|| {
                    model_config.model.file_name().unwrap().to_string_lossy().as_ref()
                });
                current_alias == model_alias
            }
            None => false,
        }
    }

    pub async fn start_with_model(&self, model_alias: &str) -> Result<bool> {
        // Find the model configuration
        let model_config = self.get_model_by_alias(model_alias).await;
        let model_config = match model_config {
            Some(cfg) => cfg,
            None => {
                error!("Model {} not found in runner {}", model_alias, self.runner_name);
                return Ok(false);
            }
        };

        // Check if this model is already loaded and runner is active
        if self.is_running().await? && self.is_model_loaded(model_alias).await {
            info!(
                "Model {} is already loaded in runner {}",
                model_alias, self.runner_name
            );
            return Ok(true);
        }

        let mut process_lock = self.process.lock().await;
        let mut is_starting_lock = self.is_starting.lock().await;

        if is_starting_lock.to_owned() {
            info!("Runner {} is already starting", self.runner_name);
            // Wait for the existing start process to complete
            drop(is_starting_lock); // Release lock before awaiting
            loop {
                sleep(Duration::from_millis(500)).await;
                let is_starting_check = self.is_starting.lock().await;
                if !is_starting_check.to_owned() {
                    break;
                }
            }
            // After loop, re-check if it successfully started
            return Ok(process_lock.is_some() && process_lock.as_mut().unwrap().try_wait()?.is_none());
        }

        // If a different model is loaded, stop the runner first
        if process_lock.is_some() && !self.is_model_loaded(model_alias).await {
            let current_alias = self.current_model.read().await.as_ref().map(|m|
                m.model_alias.as_deref().unwrap_or_else(|| m.model.file_name().unwrap().to_string_lossy().as_ref())
            ).unwrap_or("unknown");
            info!(
                "Switching runner {} from model {} to {}",
                self.runner_name, current_alias, model_alias
            );
            drop(process_lock); // Release lock before stopping
            self.stop().await?;
            process_lock = self.process.lock().await; // Re-acquire lock
        }

        // Start with the specified model
        *is_starting_lock = true;
        drop(is_starting_lock); // Release is_starting_lock early

        let result = self._start_with_specific_model(&mut process_lock, &model_config).await;
        
        *self.is_starting.lock().await = false; // Reset is_starting flag
        result
    }

    // Default start function, uses the first model assigned to this runner
    pub async fn start(&self) -> Result<bool> {
        let models_guard = self.models.read().await;
        if models_guard.is_empty() {
            error!("Runner {} has no models configured.", self.runner_name);
            return Ok(false);
        }
        let first_model_config = models_guard[0].clone();
        drop(models_guard); // Release read lock before acquiring write lock for process

        let mut process_lock = self.process.lock().await;
        let mut is_starting_lock = self.is_starting.lock().await;

        if is_starting_lock.to_owned() {
            info!("Runner {} is already starting", self.runner_name);
            drop(is_starting_lock); // Release lock before awaiting
            loop {
                sleep(Duration::from_millis(500)).await;
                let is_starting_check = self.is_starting.lock().await;
                if !is_starting_check.to_owned() {
                    break;
                }
            }
            return Ok(process_lock.is_some() && process_lock.as_mut().unwrap().try_wait()?.is_none());
        }

        *is_starting_lock = true;
        drop(is_starting_lock); // Release is_starting_lock early

        let result = self._start_with_specific_model(&mut process_lock, &first_model_config).await;
        
        *self.is_starting.lock().await = false; // Reset is_starting flag
        result
    }

    async fn _start_with_specific_model(&self, process_lock: &mut tokio::sync::MutexGuard<'_, Option<Child>>, model_config: &ModelConfig) -> Result<bool> {
        if process_lock.is_some() && process_lock.as_mut().unwrap().try_wait()?.is_none() {
            info!("Runner {} is already running", self.runner_name);
            return Ok(true);
        }

        // Build command
        let cmd = self.build_command(model_config)?;
        let log_dir = &self.session_log_dir;
        tokio::fs::create_dir_all(log_dir).await?; // Ensure log directory exists

        let log_file_path = log_dir.join(format!("{}.log", self.runner_name));
        let log_file = TokioFile::options()
            .create(true)
            .append(true) // Append to preserve logs across restarts
            .open(&log_file_path)
            .await?;
        
        let model_alias = model_config.model_alias.as_deref().unwrap_or_else(|| {
            model_config.model.file_name().unwrap().to_string_lossy().as_ref()
        });

        info!("Starting runner {} with model {}", self.runner_name, model_alias);
        debug!("Command: {:?}", cmd);
        info!("Log file: {:?}", log_file_path);

        let mut output_file_lock = self.output_file.lock().await;
        *output_file_lock = Some(log_file);
        let log_file_ref = output_file_lock.as_ref().unwrap();

        // Write separator for new start
        tokio::io::AsyncWriteExt::write_all(
            &mut log_file_ref.try_clone().await?,
            format!("\n=== Starting with model {} at {} ===\n", model_alias, chrono::Local::now().format("%Y-%m-%d %H:%M:%S")).as_bytes()
        ).await?;
        tokio::io::AsyncWriteExt::flush(&mut log_file_ref.try_clone().await?).await?;

        // Start process
        let child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .stdout(Stdio::from(log_file_ref.try_clone().await?))
            .stderr(Stdio::from(log_file_ref.try_clone().await?))
            .spawn()
            .with_context(|| format!("Failed to spawn process for runner {}", self.runner_name))?;

        *process_lock = Some(child);

        // Initial wait
        sleep(Duration::from_secs(2)).await;

        // Check if process is still running after initial wait
        if process_lock.as_mut().unwrap().try_wait()?.is_some() {
            error!(
                "Runner {} exited immediately with code: {:?}",
                self.runner_name, process_lock.as_ref().unwrap().try_wait()?.unwrap().code()
            );
            self.cleanup_runner_state(process_lock).await;
            return Ok(false);
        }

        // Wait for server to be ready
        let max_retries = 30;
        let retry_interval_ms = 1000;
        for i in 0..max_retries {
            if self._is_server_ready().await {
                info!(
                    "Runner {} started successfully with model {}",
                    self.runner_name, model_alias
                );
                *self.current_model.write().await = Some(model_config.clone());
                return Ok(true);
            }

            // Check if process has exited during readiness check
            if process_lock.as_mut().unwrap().try_wait()?.is_some() {
                error!(
                    "Runner {} exited during readiness check with code: {:?}",
                    self.runner_name, process_lock.as_ref().unwrap().try_wait()?.unwrap().code()
                );
                self.cleanup_runner_state(process_lock).await;
                return Ok(false);
            }

            debug!(
                "Runner {} not ready yet, retrying in {}ms (attempt {}/{})",
                self.runner_name, retry_interval_ms, i + 1, max_retries
            );
            sleep(Duration::from_millis(retry_interval_ms)).await;
        }

        // Server did not start in time
        error!("Runner {} did not start in time", self.runner_name);
        self.stop().await?;
        Ok(false)
    }

    pub async fn stop(&self) -> Result<bool> {
        let mut process_lock = self.process.lock().await;
        if process_lock.is_none() {
            info!("Runner {} is not running", self.runner_name);
            return Ok(true);
        }

        let current_alias = self.current_model.read().await.as_ref().map(|m|
            m.model_alias.as_deref().unwrap_or_else(|| m.model.file_name().unwrap().to_string_lossy().as_ref())
        ).unwrap_or("unknown");
        info!(
            "Stopping runner {} (current model: {})",
            self.runner_name, current_alias
        );

        // Try graceful termination first
        if let Some(child) = process_lock.as_mut() {
            child.start_kill()?;
        }

        // Wait for process to terminate
        for _ in 0..50 {
            // 5 seconds with 0.1s intervals
            if process_lock.as_mut().unwrap().try_wait()?.is_some() {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }

        // If still running, force kill
        if process_lock.as_mut().unwrap().try_wait()?.is_none() {
            warn!("Runner {} did not terminate gracefully, forcing kill", self.runner_name);
            if let Some(child) = process_lock.as_mut() {
                child.kill().await?;
            }
            // Wait again
            for _ in 0..50 {
                if process_lock.as_mut().unwrap().try_wait()?.is_some() {
                    break;
                }
                sleep(Duration::from_millis(100)).await;
            }
        }
        
        let exit_code = process_lock.as_ref().unwrap().try_wait()?.map(|s| s.code()).flatten();
        info!("Runner {} stopped with exit code: {:?}", self.runner_name, exit_code);

        self.cleanup_runner_state(process_lock).await;

        // Wait to offload GPU memory (small delay)
        sleep(Duration::from_millis(500)).await;

        Ok(true)
    }

    async fn cleanup_runner_state(&self, mut process_lock: tokio::sync::MutexGuard<'_, Option<Child>>) {
        // Close log file
        let mut output_file_lock = self.output_file.lock().await;
        if let Some(file) = output_file_lock.take() {
            drop(file); // Drops the TokioFile, closing it
        }

        // Reset process and current_model
        *process_lock = None;
        *self.current_model.write().await = None;
    }

    pub async fn is_running(&self) -> Result<bool> {
        let mut process_lock = self.process.lock().await;
        if let Some(child) = process_lock.as_mut() {
            if let Some(status) = child.try_wait()? {
                let current_alias = self.current_model.read().await.as_ref().map(|m|
                    m.model_alias.as_deref().unwrap_or_else(|| m.model.file_name().unwrap().to_string_lossy().as_ref())
                ).unwrap_or("unknown");
                warn!(
                    "Runner {} has exited with code: {:?} (was running model: {})",
                    self.runner_name, status.code(), current_alias
                );
                self.cleanup_runner_state(process_lock).await;
                Ok(false)
            } else {
                Ok(true)
            }
        } else {
            Ok(false)
        }
    }

    fn build_command(&self, model_config: &ModelConfig) -> Result<Vec<String>> {
        let mut cmd = Vec::new();
        let runner_path = self.runner_config.path.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Runner {} has no executable path defined.", self.runner_name)
        })?;
        cmd.push(runner_path.to_string_lossy().into_owned());

        cmd.push("--model".to_string());
        cmd.push(model_config.model.to_string_lossy().into_owned());

        cmd.push("--host".to_string());
        cmd.push(self.host.clone());
        cmd.push("--port".to_string());
        cmd.push(self.port.to_string());

        if let Some(mmproj) = &model_config.mmproj {
            cmd.push("--mmproj".to_string());
            cmd.push(mmproj.to_string_lossy().into_owned());
        }
        if let Some(alias) = &model_config.model_alias {
            cmd.push("--alias".to_string());
            cmd.push(alias.clone());
        }
        if let Some(n_ctx) = model_config.n_ctx {
            cmd.push("--ctx-size".to_string());
            cmd.push(n_ctx.to_string());
        }
        if let Some(n_batch) = model_config.n_batch {
            cmd.push("--batch-size".to_string());
            cmd.push(n_batch.to_string());
        }
        if let Some(n_threads) = model_config.n_threads {
            cmd.push("--threads".to_string());
            cmd.push(n_threads.to_string());
        }
        if let Some(chat_template) = &model_config.chat_template {
            cmd.push("--chat-template".to_string());
            cmd.push(chat_template.clone());
        }
        if let Some(split_mode) = &model_config.split_mode {
            cmd.push("--split-mode".to_string());
            cmd.push(split_mode.clone());
        }
        if model_config.embedding {
            cmd.push("--embedding".to_string());
        }
        if model_config.reranking {
            cmd.push("--reranking".to_string());
        }
        if !model_config.offload_kqv {
            cmd.push("--no-kv-offload".to_string());
        }
        if model_config.jinja {
            cmd.push("--jinja".to_string());
        }
        if let Some(pooling) = &model_config.pooling {
            cmd.push("--pooling".to_string());
            cmd.push(pooling.clone());
        }
        if model_config.flash_attn {
            cmd.push("--flash-attn".to_string());
        }
        if model_config.use_mlock {
            cmd.push("--mlock".to_string());
        }
        if let Some(main_gpu) = model_config.main_gpu {
            cmd.push("--main-gpu".to_string());
            cmd.push(main_gpu.to_string());
        }
        if !model_config.tensor_split.is_empty() {
            cmd.push("--tensor-split".to_string());
            cmd.push(model_config.tensor_split.iter().map(|f| f.to_string()).collect::<Vec<String>>().join(","));
        }
        if let Some(n_gpu_layers) = model_config.n_gpu_layers {
            cmd.push("--n-gpu-layers".to_string());
            cmd.push(n_gpu_layers.to_string());
        }
        if let Some(cache_k) = &model_config.cache_type_k {
            cmd.push("--cache-type-k".to_string());
            cmd.push(cache_k.clone());
        }
        if let Some(cache_v) = &model_config.cache_type_v {
            cmd.push("--cache-type-v".to_string());
            cmd.push(cache_v.clone());
        }
        if let Some(rope_scaling) = &model_config.rope_scaling {
            cmd.push("--rope-scaling".to_string());
            cmd.push(rope_scaling.clone());
        }
        if let Some(rope_scale) = model_config.rope_scale {
            cmd.push("--rope-scale".to_string());
            cmd.push(rope_scale.to_string());
        }
        if let Some(yarn_orig_ctx) = model_config.yarn_orig_ctx {
            cmd.push("--yarn-orig-ctx".to_string());
            cmd.push(yarn_orig_ctx.to_string());
        }

        cmd.extend(self.runner_config.extra_args.clone());

        Ok(cmd)
    }

    async fn _is_server_ready(&self) -> bool {
        let health_url = format!("http://{}:{}/health", self.host, self.port);
        match self.reqwest_client.get(&health_url).timeout(Duration::from_secs(1)).send().await {
            Ok(response) => response.status().is_success(),
            Err(e) => {
                debug!("Health check for {}:{} failed: {:?}", self.host, self.port, e);
                false
            }
        }
    }

    pub async fn get_current_model_alias(&self) -> Option<String> {
        self.current_model.read().await.as_ref().map(|m| {
            m.model_alias.clone().unwrap_or_else(|| {
                m.model.file_name().unwrap().to_string_lossy().into_owned()
            })
        })
    }
}

/// Manager for FlexLLama processes.
pub struct RunnerManager {
    config_manager: Arc<ConfigManager>,
    pub runners: HashMap<String, Arc<RunnerProcess>>,
    model_runner_map: HashMap<String, String>, // model_alias -> runner_name
    session_log_dir: PathBuf,
    reqwest_client: Client,
}

impl RunnerManager {
    pub fn new(config_manager: Arc<ConfigManager>, session_log_dir: PathBuf) -> Result<Self> {
        let mut runners = HashMap::new();
        let mut model_runner_map = HashMap::new();

        // Initialize runner processes
        for runner_name in config_manager.get_runner_names() {
            let runner_config = config_manager.get_runner_config(&runner_name)
                .ok_or_else(|| anyhow::anyhow!("Runner config for {} not found", runner_name))?;
            
            // Resolve runner host and port, falling back to API config or auto-assignment later
            let host = runner_config.host.clone().unwrap_or_else(|| config_manager.get_api_host().to_string());
            let port = runner_config.port.unwrap_or_else(|| {
                // In a real implementation, you'd have a more robust auto-port assignment logic
                warn!("Runner '{}': 'port' not specified, a fixed default or dynamic assignment should be implemented here.", runner_name);
                // For now, let's pick an arbitrary default or error if not found in config
                // This is a placeholder, a proper dynamic port allocation would be complex here.
                // For now, if config doesn't provide port, this will be a problem later.
                // Assuming all runner ports are explicitly defined in config_example.json
                0 // This should ideally be dynamically assigned or an error if not in config
            });

            runners.insert(
                runner_name.clone(),
                Arc::new(RunnerProcess::new(
                    runner_name.clone(),
                    runner_config.clone(),
                    host,
                    port,
                    session_log_dir.clone(),
                )),
            );
        }

        // Assign models to runners
        for model_config in config_manager.get_config().models.iter() {
            let model_alias = model_config.model_alias.clone().unwrap_or_else(|| {
                model_config.model.file_name().unwrap().to_string_lossy().into_owned()
            });
            let runner_name = model_config.runner.clone();

            if let Some(runner_process) = runners.get(&runner_name) {
                // Add model to the runner. This is an async operation, but `new` is sync.
                // We'll queue these up to be added after construction, or adjust RunnerProcess::new to be async
                // For simplicity now, we'll make add_model async later and call it after construction
                // Alternatively, RunnerProcess could take an initial list of models.
                // For now, let's keep model assignment simple for constructor and add async_trait for add_model
            } else {
                error!(
                    "Model {} references unknown runner {}",
                    model_alias, runner_name
                );
            }
            model_runner_map.insert(model_alias, runner_name);
        }
        
        let manager = Self {
            config_manager,
            runners,
            model_runner_map,
            session_log_dir,
            reqwest_client: Client::new(),
        };

        // Now, asynchronously add models to runners after all runners are initialized
        // This requires an async context, so we'll do it from `main` or a separate `init` method.
        // For now, the models field in RunnerProcess remains empty after construction.
        // A better design might be to pass models during RunnerProcess construction,
        // or have an async init method for RunnerManager.

        Ok(manager)
    }

    pub async fn init_models_for_runners(&self) -> Result<()> {
        for model_config in self.config_manager.get_config().models.iter() {
            let runner_name = &model_config.runner;
            if let Some(runner_process) = self.runners.get(runner_name) {
                runner_process.add_model(model_config.clone()).await;
            }
        }
        Ok(())
    }

    pub async fn start_runner(&self, runner_name: &str) -> Result<bool> {
        match self.runners.get(runner_name) {
            Some(runner) => runner.start().await,
            None => {
                error!("Unknown runner: {}", runner_name);
                Ok(false)
            }
        }
    }

    pub async fn start_runner_for_model(&self, model_alias: &str) -> Result<bool> {
        match self.model_runner_map.get(model_alias) {
            Some(runner_name) => {
                let runner = self.runners.get(runner_name).context(format!("Runner {} not found in map", runner_name))?;
                runner.start_with_model(model_alias).await
            },
            None => {
                error!("Unknown model: {}", model_alias);
                Ok(false)
            }
        }
    }

    pub async fn stop_runner(&self, runner_name: &str) -> Result<bool> {
        match self.runners.get(runner_name) {
            Some(runner) => runner.stop().await,
            None => {
                error!("Unknown runner: {}", runner_name);
                Ok(false)
            }
        }
    }

    pub async fn stop_all_runners(&self) -> Result<bool> {
        info!("Stopping all runners...");
        let mut success = true;
        for runner in self.runners.values() {
            if !runner.stop().await? {
                success = false;
            }
        }
        Ok(success)
    }

    pub async fn auto_start_default_runners(&self) -> Result<bool> {
        if !self.config_manager.get_auto_start_runners() {
            info!("Auto-start is disabled, skipping runner auto-start");
            return Ok(true);
        }

        info!("Auto-starting default runners...");
        let mut success = true;
        let mut started_count = 0;

        for (runner_name, runner) in &self.runners {
            let models_for_runner = runner.models.read().await;
            if !models_for_runner.is_empty() {
                let first_model_alias = models_for_runner[0].model_alias.clone().unwrap_or_else(|| {
                    models_for_runner[0].model.file_name().unwrap().to_string_lossy().into_owned()
                });
                info!(
                    "Auto-starting runner {} with model {}",
                    runner_name, first_model_alias
                );
                if runner.start().await? {
                    started_count += 1;
                    info!("Successfully auto-started runner {}", runner_name);
                } else {
                    error!("Failed to auto-start runner {}", runner_name);
                    success = false;
                }
            } else {
                warn!(
                    "Runner {} has no models assigned, skipping auto-start",
                    runner_name
                );
            }
        }

        if success && started_count > 0 {
            info!("Successfully auto-started {} runners", started_count);
        } else if started_count == 0 {
            info!("No runners were auto-started (no models assigned)");
        }

        Ok(success)
    }

    pub async fn is_runner_running(&self, runner_name: &str) -> Result<bool> {
        match self.runners.get(runner_name) {
            Some(runner) => runner.is_running().await,
            None => {
                error!("Unknown runner: {}", runner_name);
                Ok(false)
            }
        }
    }

    pub async fn is_model_available(&self, model_alias: &str) -> Result<bool> {
        match self.model_runner_map.get(model_alias) {
            Some(runner_name) => {
                let runner = self.runners.get(runner_name).context(format!("Runner {} not found in map", runner_name))?;
                if !runner.is_running().await? {
                    return Ok(false);
                }
                Ok(runner.is_model_loaded(model_alias).await)
            }
            None => {
                error!("Unknown model: {}", model_alias);
                Ok(false)
            }
        }
    }

    pub fn get_runner_for_model(&self, model_alias: &str) -> Option<Arc<RunnerProcess>> {
        self.model_runner_map.get(model_alias)
            .and_then(|runner_name| self.runners.get(runner_name).cloned())
    }

    pub fn get_model_aliases(&self) -> Vec<String> {
        self.config_manager.get_model_aliases()
    }

    pub fn get_runner_names(&self) -> Vec<String> {
        self.config_manager.get_runner_names()
    }

    pub fn get_model_runner_map(&self) -> HashMap<String, String> {
        self.model_runner_map.clone()
    }

    pub async fn get_current_model_for_runner(&self, runner_name: &str) -> Option<String> {
        self.runners.get(runner_name)
            .and_then(|runner| Some(runner.get_current_model_alias().await?))
    }

    pub async fn switch_model(&self, from_model_alias: &str, to_model_alias: &str) -> Result<bool> {
        let from_runner_name = self.model_runner_map.get(from_model_alias)
            .ok_or_else(|| anyhow::anyhow!("Unknown source model: {}", from_model_alias))?;
        let to_runner_name = self.model_runner_map.get(to_model_alias)
            .ok_or_else(|| anyhow::anyhow!("Unknown target model: {}", to_model_alias))?;

        if from_runner_name != to_runner_name {
            bail!(
                "Models {} and {} are on different runners ({} vs {})",
                from_model_alias, to_model_alias, from_runner_name, to_runner_name
            );
        }

        self.start_runner_for_model(to_model_alias).await
    }

    pub async fn get_runner_status(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut status = HashMap::new();
        for (runner_name, runner) in &self.runners {
            let is_running = runner.is_running().await?;
            let current_model = runner.get_current_model_alias().await;
            let available_models = runner.models.read().await.iter().map(|m| {
                m.model_alias.clone().unwrap_or_else(|| m.model.file_name().unwrap().to_string_lossy().into_owned())
            }).collect::<Vec<String>>();

            status.insert(runner_name.clone(), serde_json::json!({
                "is_running": is_running,
                "current_model": current_model,
                "available_models": available_models,
                "host": runner.host,
                "port": runner.port,
            }));
        }
        Ok(status)
    }

    pub async fn ensure_model_ready_with_retry(&self, model_alias: &str) -> (bool, Option<String>) {
        let max_retries = self.config_manager.get_max_retries();
        let base_delay = self.config_manager.get_base_delay_seconds();
        let max_delay = self.config_manager.get_max_delay_seconds();
        let retry_on_loading = self.config_manager.get_retry_on_model_loading();

        let mut last_error = None;

        for attempt in 0..=max_retries {
            if attempt > 0 {
                let delay = Duration::from_secs(std::cmp::min(base_delay * (2u64.pow((attempt - 1) as u32)), max_delay));
                info!(
                    "Retrying model readiness check for {} (attempt {}/{}) after {:?} delay",
                    model_alias, attempt + 1, max_retries + 1, delay
                );
                sleep(delay).await;
            } else {
                debug!("Checking model readiness for {}", model_alias);
            }

            // Ensure model is started
            if !self.is_model_available(model_alias).await.unwrap_or(false) {
                info!("Starting runner for model {}", model_alias);
                if !self.start_runner_for_model(model_alias).await.unwrap_or(false) {
                    last_error = Some(format!("Failed to start model: {}", model_alias));
                    if attempt == max_retries {
                        break;
                    }
                    continue;
                }
            }

            // Wait for the newly started model to become ready
            debug!("Waiting for newly started model {} to become ready", model_alias);
            self._wait_for_model_readiness(model_alias, 30).await;

            // Do a pre-flight readiness check
            let (is_ready, readiness_error) = self._check_model_readiness(model_alias).await;
            if !is_ready {
                last_error = readiness_error.clone();
                if let Some(err_msg) = &readiness_error {
                    info!("Model {} not ready: {}", model_alias, err_msg);
                }
                if attempt < max_retries && (retry_on_loading || readiness_error.as_deref() != Some(HealthMessages::MODEL_LOADING)) {
                    // Only continue retry if retry_on_loading is true or it's not a loading error
                    continue;
                } else {
                    break;
                }
            }

            // Model is ready
            debug!("Model {} is ready", model_alias);
            return (true, None);
        }

        // All retries exhausted
        error!(
            "Model readiness check for {} failed after {} attempts. Last error: {:?}",
            model_alias, max_retries + 1, last_error
        );
        (false, last_error)
    }

    async fn _check_model_readiness(&self, model_alias: &str) -> (bool, Option<String>) {
        let runner = match self.get_runner_for_model(model_alias) {
            Some(r) => r,
            None => return (false, Some(HealthMessages::NO_RUNNER_AVAILABLE.to_string())),
        };

        let health_url = format!("http://{}:{}/health", runner.host, runner.port);
        match self.reqwest_client.get(&health_url).timeout(Duration::from_secs(5)).send().await {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    (true, None)
                } else if status == StatusCode::SERVICE_UNAVAILABLE { // 503 Service Unavailable
                    match response.json::<serde_json::Value>().await {
                        Ok(data) => {
                            let error_message = data["error"]["message"].as_str().unwrap_or("Unknown error").to_string();
                            if error_message.to_lowercase().contains("loading") {
                                (false, Some(HealthMessages::MODEL_LOADING.to_string()))
                            } else {
                                (false, Some(error_message))
                            }
                        },
                        Err(_) => {
                            let text = response.text().await.unwrap_or_default();
                            if text.to_lowercase().contains("loading") {
                                (false, Some(HealthMessages::MODEL_LOADING.to_string()))
                            } else {
                                (false, Some(format!("HTTP {}: {}", status, text)))
                            }
                        }
                    }
                } else {
                    let text = response.text().await.unwrap_or_default();
                    (false, Some(format!("Health check failed with status {}: {}", status, text)))
                }
            },
            Err(e) => {
                if e.is_timeout() {
                    (false, Some(HealthMessages::HEALTH_CHECK_TIMEOUT.to_string()))
                } else {
                    (false, Some(format!("{}: {}", HealthMessages::CONNECTION_ERROR, e)))
                }
            }
        }
    }

    async fn _wait_for_model_readiness(&self, model_alias: &str, max_wait_seconds: u64) {
        let start_time = tokio::time::Instant::now();
        while (tokio::time::Instant::now() - start_time).as_secs() < max_wait_seconds {
            let (is_ready, _) = self._check_model_readiness(model_alias).await;
            if is_ready {
                debug!(
                    "Model {} became ready after {:.1}s",
                    model_alias, (tokio::time::Instant::now() - start_time).as_secs_f32()
                );
                return;
            }
            sleep(Duration::from_millis(500)).await;
        }
        warn!(
            "Model {} did not become ready within {}s",
            model_alias, max_wait_seconds
        );
    }

    pub async fn forward_request(
        &self,
        model_alias: &str,
        endpoint: &str,
        request_data: serde_json::Value,
        timeout_secs: u64,
    ) -> (bool, serde_json::Value, StatusCode) {
        let runner = match self.get_runner_for_model(model_alias) {
            Some(r) => r,
            None => {
                return (
                    false,
                    serde_json::json!({"error": {"message": format!("Model not available: {}", model_alias)}}),
                    StatusCode::INTERNAL_SERVER_ERROR,
                );
            }
        };

        let url = format!("http://{}{}", runner.port, endpoint); // Use runner.host and runner.port

        match self.reqwest_client.post(&url)
            .json(&request_data)
            .timeout(Duration::from_secs(timeout_secs))
            .send()
            .await
        {
            Ok(response) => {
                let status = response.status();
                let response_data = match response.json::<serde_json::Value>().await {
                    Ok(data) => data,
                    Err(e) => {
                        error!("Failed to parse response from {}: {}", url, e);
                        serde_json::json!({"error": {"message": format!("Invalid response from upstream: {}", e)}})
                    }
                };
                (status.is_success(), response_data, status)
            }
            Err(e) => {
                error!("Client error forwarding to {}: {}", url, e);
                let status_code = if e.is_timeout() {
                    StatusCode::REQUEST_TIMEOUT // 408
                } else {
                    StatusCode::SERVICE_UNAVAILABLE // 503
                };
                (
                    false,
                    serde_json::json!({"error": {"message": format!("Connection error: {}", e)}}),
                    status_code,
                )
            }
        }
    }

    pub async fn check_model_health(&self, model_alias: &str) -> serde_json::Value {
        let (is_ready, error_message) = self._check_model_readiness(model_alias).await;
        serde_json::json!({
            "status": if is_ready { HealthStatus::Ok.as_str() } else { HealthStatus::Error.as_str() },
            "message": if is_ready { HealthMessages::READY } else { error_message.unwrap_or_else(|| HealthMessages::HEALTH_CHECK_FAILED.to_string()) },
            "model_alias": model_alias,
        })
    }
}