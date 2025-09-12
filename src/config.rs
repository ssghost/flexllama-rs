// src/config.rs

use std::{collections::HashMap, path::PathBuf};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use tokio::fs::File as TokioFile;
use tokio::io::AsyncReadExt;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ModelConfig {
    pub model: PathBuf,
    pub model_alias: Option<String>,
    pub mmproj: Option<PathBuf>,
    pub runner: String,
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_threads: Option<u32>,
    pub chat_template: Option<String>,
    pub split_mode: Option<String>,
    pub embedding: bool,
    pub reranking: bool,
    pub offload_kqv: bool,
    pub jinja: bool,
    pub pooling: Option<String>,
    pub flash_attn: bool,
    pub use_mlock: bool,
    pub main_gpu: Option<u32>,
    pub tensor_split: Vec<f32>,
    pub n_gpu_layers: Option<i32>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
    pub rope_scaling: Option<String>,
    pub rope_scale: Option<f32>,
    pub yarn_orig_ctx: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RunnerConfig {
    pub path: Option<PathBuf>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Config {
    pub api_host: String,
    pub api_port: u16,
    pub auto_start_runners: bool,
    pub max_retries: u64,
    pub base_delay_seconds: u64,
    pub max_delay_seconds: u64,
    pub retry_on_model_loading: bool,
    pub models: Vec<ModelConfig>,
    pub runners: HashMap<String, RunnerConfig>,
}

pub struct ConfigManager {
    config: Config,
    config_path: PathBuf,
}

impl ConfigManager {
    pub async fn new(config_path: PathBuf) -> Result<Self> {
        let config_str = Self::load_config_file(&config_path)
            .await
            .context("Failed to load configuration file")?;
        
        let config: Config = serde_json::from_str(&config_str)
            .context("Failed to deserialize configuration")?;

        Ok(Self {
            config,
            config_path,
        })
    }

    async fn load_config_file(path: &PathBuf) -> Result<String> {
        let mut file = TokioFile::open(path).await?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).await?;
        Ok(contents)
    }

    pub fn get_config(&self) -> &Config {
        &self.config
    }

    pub fn get_api_host(&self) -> &str {
        &self.config.api_host
    }

    pub fn get_api_port(&self) -> u16 {
        self.config.api_port
    }

    pub fn get_auto_start_runners(&self) -> bool {
        self.config.auto_start_runners
    }

    pub fn get_max_retries(&self) -> u64 {
        self.config.max_retries
    }

    pub fn get_base_delay_seconds(&self) -> u64 {
        self.config.base_delay_seconds
    }

    pub fn get_max_delay_seconds(&self) -> u64 {
        self.config.max_delay_seconds
    }

    pub fn get_retry_on_model_loading(&self) -> bool {
        self.config.retry_on_model_loading
    }

    pub fn get_runner_names(&self) -> Vec<String> {
        self.config.runners.keys().cloned().collect()
    }

    pub fn get_runner_config(&self, runner_name: &str) -> Option<&RunnerConfig> {
        self.config.runners.get(runner_name)
    }

    pub fn get_model_aliases(&self) -> Vec<String> {
        self.config.models.iter().map(|m| {
            m.model_alias.clone().unwrap_or_else(|| {
                m.model.file_name().unwrap().to_string_lossy().into_owned()
            })
        }).collect()
    }
}