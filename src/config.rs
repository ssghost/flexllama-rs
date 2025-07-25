// src/config.rs

use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};
use serde::{Deserialize, Serialize};
use anyhow::{Result, bail};
use tracing::{info, error, warn};

/// Top-level application configuration.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    #[serde(default = "default_auto_start_runners")]
    pub auto_start_runners: bool,
    pub api: ApiConfig,
    #[serde(default)]
    pub retry_config: RetryConfig,
    // Use a HashMap to store runners by their arbitrary names (e.g., "runner1")
    #[serde(flatten)]
    pub runners: HashMap<String, RunnerConfig>,
    pub models: Vec<ModelConfig>,
}

fn default_auto_start_runners() -> bool {
    true
}

/// Configuration for the API server.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
}

/// Configuration for retry logic when models are loading.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RetryConfig {
    #[serde(default = "default_max_retries")]
    pub max_retries: u8,
    #[serde(default = "default_base_delay_seconds")]
    pub base_delay_seconds: u64,
    #[serde(default = "default_max_delay_seconds")]
    pub max_delay_seconds: u64,
    #[serde(default = "default_retry_on_model_loading")]
    pub retry_on_model_loading: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            base_delay_seconds: default_base_delay_seconds(),
            max_delay_seconds: default_max_delay_seconds(),
            retry_on_model_loading: default_retry_on_model_loading(),
        }
    }
}

fn default_max_retries() -> u8 { 5 }
fn default_base_delay_seconds() -> u64 { 2 }
fn default_max_delay_seconds() -> u64 { 30 }
fn default_retry_on_model_loading() -> bool { true }

/// Configuration for a llama.cpp runner process.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RunnerConfig {
    #[serde(rename = "type")]
    pub runner_type: String, // e.g., "llama-server"
    pub path: Option<PathBuf>, // Path to llama-server binary, if not default
    pub host: Option<String>,
    pub port: Option<u16>,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

/// Configuration for a specific AI model.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelConfig {
    pub runner: String, // Name of the runner this model belongs to
    pub model: PathBuf, // Path to .gguf model file
    pub model_alias: Option<String>, // User-friendly name for API calls
    
    // Performance & Memory
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_threads: Option<u32>,
    pub main_gpu: Option<u8>,
    pub n_gpu_layers: Option<i32>, // Can be -1 for all, 0 for none, or specific number
    #[serde(default)]
    pub tensor_split: Vec<f32>,

    // Model Types
    #[serde(default)]
    pub embedding: bool,
    #[serde(default)]
    pub reranking: bool,
    pub mmproj: Option<PathBuf>, // For vision models

    // Optimization
    #[serde(default)]
    pub offload_kqv: bool,
    #[serde(default)]
    pub flash_attn: bool,
    #[serde(default)]
    pub use_mlock: bool,
    pub split_mode: Option<String>,
    #[serde(rename = "cache-type-k")]
    pub cache_type_k: Option<String>,
    #[serde(rename = "cache-type-v")]
    pub cache_type_v: Option<String>,

    // Chat & Templates
    pub chat_template: Option<String>,
    #[serde(default)]
    pub jinja: bool,

    // Advanced Options
    #[serde(rename = "rope-scaling")]
    pub rope_scaling: Option<String>,
    #[serde(rename = "rope-scale")]
    pub rope_scale: Option<f32>,
    #[serde(rename = "yarn-orig-ctx")]
    pub yarn_orig_ctx: Option<u32>,
    pub pooling: Option<String>,
}

pub struct ConfigManager {
    config: Config,
    config_path: PathBuf,
}

impl ConfigManager {
    pub fn new(config_path: PathBuf) -> Result<Self> {
        info!("Loading configuration from {:?}", config_path);
        let config = Self::load_config(&config_path)?;
        let mut manager = Self { config, config_path };
        manager.validate_and_normalize_config()?;
        Ok(manager)
    }

    fn load_config(path: &PathBuf) -> Result<Config> {
        if !path.exists() {
            bail!("Configuration file not found: {:?}", path);
        }
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let config: Config = serde_json::from_str(&contents)?;
        Ok(config)
    }

    // This method will perform comprehensive validation and normalization,
    // similar to Python's _validate_config.
    fn validate_and_normalize_config(&mut self) -> Result<()> {
        info!("Validating configuration...");

        // Validate API configuration (already done by serde, but ensure defaults/logic)
        // No explicit validation needed here beyond deserialization success for basic fields.

        // Validate retry configuration
        if self.config.retry_config.max_delay_seconds < self.config.retry_config.base_delay_seconds {
            bail!("retry_config: max_delay_seconds must be greater than or equal to base_delay_seconds");
        }

        let mut used_ports = std::collections::HashSet::new();

        // Validate runners and collect their names
        let mut runner_names: std::collections::HashSet<String> = self.config.runners.keys().cloned().collect();

        // If a runner's path is not specified, default it to its type (like in Python)
        for (runner_name, runner_config) in self.config.runners.iter_mut() {
            if runner_config.path.is_none() {
                runner_config.path = Some(PathBuf::from(&runner_config.runner_type));
                warn!("Runner '{}': 'path' not specified, defaulting to '{}'", runner_name, runner_config.runner_type);
            }

            // Validate port conflicts for runners
            if let Some(port) = runner_config.port {
                if !used_ports.insert(port) {
                    bail!("Runner '{}': Port {} already in use by another runner.", runner_name, port);
                }
            }
            // TODO: Auto-assign ports if not provided, mimicking Python's behavior,
            // or ensure all runners have explicit ports for now.
            // For now, if no port is provided, it will be None.
        }

        // Validate models
        if self.config.models.is_empty() {
            bail!("Configuration must contain at least one model.");
        }

        for (i, model) in self.config.models.iter().enumerate() {
            if !runner_names.contains(&model.runner) {
                bail!("Model {}: Referenced runner '{}' not found in configuration.", i, model.runner);
            }

            // Ensure model path exists for comprehensive validation later, or just check format now.
            // For now, just type check. Path existence check is more for runtime.
        }

        info!("Configuration validation successful.");
        Ok(())
    }

    pub fn get_config(&self) -> &Config {
        &self.config
    }

    pub fn get_api_host(&self) -> &str {
        &self.config.api.host
    }

    pub fn get_api_port(&self) -> u16 {
        self.config.api.port
    }

    pub fn get_model_aliases(&self) -> Vec<String> {
        self.config.models.iter().map(|m| {
            m.model_alias.clone().unwrap_or_else(|| {
                m.model.file_name().unwrap().to_string_lossy().into_owned()
            })
        }).collect()
    }

    pub fn get_runner_names(&self) -> Vec<String> {
        self.config.runners.keys().cloned().collect()
    }

    pub fn get_runner_config(&self, runner_name: &str) -> Option<&RunnerConfig> {
        self.config.runners.get(runner_name)
    }

    pub fn get_model_config(&self, model_alias: &str) -> Option<&ModelConfig> {
        self.config.models.iter().find(|m| {
            let alias = m.model_alias.clone().unwrap_or_else(|| {
                m.model.file_name().unwrap().to_string_lossy().into_owned()
            });
            alias == model_alias
        })
    }

    pub fn get_runner_for_model(&self, model_alias: &str) -> Result<(&String, &RunnerConfig)> {
        let model = self.get_model_config(model_alias)
            .ok_or_else(|| anyhow::anyhow!("Model alias not found: {}", model_alias))?;
        let runner_name = &model.runner;
        let runner_config = self.get_runner_config(runner_name)
            .ok_or_else(|| anyhow::anyhow!("Referenced runner '{}' for model '{}' not found.", runner_name, model_alias))?;
        Ok((runner_name, runner_config))
    }

    pub fn get_auto_start_runners(&self) -> bool {
        self.config.auto_start_runners
    }

    // Getter for retry config values
    pub fn get_max_retries(&self) -> u8 {
        self.config.retry_config.max_retries
    }

    pub fn get_base_delay_seconds(&self) -> u64 {
        self.config.retry_config.base_delay_seconds
    }

    pub fn get_max_delay_seconds(&self) -> u64 {
        self.config.retry_config.max_delay_seconds
    }

    pub fn get_retry_on_model_loading(&self) -> bool {
        self.config.retry_config.retry_on_model_loading
    }
}