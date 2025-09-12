// src/main.rs

use std::{path::PathBuf, io, rc::Rc, sync::Arc, collections::HashMap};
use tokio::sync::{Mutex, RwLock}; // Fix for 'sync is private'
use ratatui::{
    layout::Alignment,
    style::{Color, Stylize},
    widgets::{Block, BorderType, Paragraph},
    Frame, Terminal,
};
use ratzilla::{
    event::{KeyCode, KeyEvent},
    DomBackend, WebRenderer,
};
use anyhow::Result;
use tracing::{info, Level, error};
use tracing_subscriber::{FmtSubscriber, prelude::*};
use serde_json;

mod api;
mod config;
mod runner;

use crate::config::ConfigManager;
use crate::runner::RunnerManager;

// Shared application state that can be updated asynchronously
#[derive(Default)]
struct SharedAppState {
    runner_status: RwLock<HashMap<String, serde_json::Value>>,
    status_text: Mutex<String>,
    selected_runner: Mutex<Option<String>>,
    last_error: Mutex<Option<String>>,
}

// The main application struct for the synchronous TUI
struct App {
    shared_state: Arc<SharedAppState>,
    runner_manager: Arc<RunnerManager>,
}

impl App {
    fn render(&self, frame: &mut Frame) {
        // We use block_on here because render is a synchronous function
        // but needs to read state updated asynchronously.
        let runner_status = tokio::runtime::Handle::current().block_on(self.shared_state.runner_status.read()).clone();
        let status_text = tokio::runtime::Handle::current().block_on(self.shared_state.status_text.lock()).clone();
        let selected = tokio::runtime::Handle::current().block_on(self.shared_state.selected_runner.lock()).clone();
        let error = tokio::runtime::Handle::current().block_on(self.shared_state.last_error.lock()).clone();

        let block = Block::bordered()
            .title("FlexLLama TUI")
            .title_alignment(Alignment::Center)
            .border_type(BorderType::Rounded);

        let mut status_lines = vec![
            format!("Status: {}", status_text),
        ];

        if let Some(err) = error {
            status_lines.push(format!("Error: {}", err).fg(Color::Red).to_string());
        }

        status_lines.push("".to_string());

        for (runner_name, runner_info) in runner_status {
            let is_running = runner_info["is_running"].as_bool().unwrap_or(false);
            let current_model = runner_info["current_model"].as_str().unwrap_or("None");
            let line_color = if is_running { Color::Green } else { Color::DarkGray };
            let prefix = if selected.as_ref().map_or(false, |s| s == &runner_name) { "▶ " } else { "  " };
            
            status_lines.push(format!("{}Runner: {} | Running: {} | Model: {}", prefix, runner_name, is_running, current_model).fg(line_color).to_string());
        }

        status_lines.push("".to_string());
        status_lines.push("Press 'S' to start, 'O' to stop, 'Q' to quit.".to_string());

        let paragraph = Paragraph::new(status_lines.join("\n"))
            .block(block)
            .fg(Color::White)
            .bg(Color::Black)
            .centered();

        frame.render_widget(paragraph, frame.area());
    }

    fn handle_events(&self, key_event: KeyEvent) {
        let shared_state = Arc::clone(&self.shared_state);
        let runner_manager = Arc::clone(&self.runner_manager);

        tokio::spawn(async move {
            let mut last_error = shared_state.last_error.lock().await;
            last_error.take();

            let mut selected = shared_state.selected_runner.lock().await;

            match key_event.code {
                KeyCode::Char('s') => {
                    let mut status = shared_state.status_text.lock().await;
                    if let Some(runner_name) = selected.as_ref() {
                        *status = format!("Starting runner: {}", runner_name);
                        info!("User requested start for runner: {}", runner_name);
                        if let Err(e) = runner_manager.start_runner(runner_name).await {
                            *last_error = Some(format!("Failed to start runner: {}", e));
                        }
                    } else {
                        *last_error = Some("No runner selected. Please select a runner first.".to_string());
                    }
                }
                KeyCode::Char('o') => {
                    let mut status = shared_state.status_text.lock().await;
                    if let Some(runner_name) = selected.as_ref() {
                        *status = format!("Stopping runner: {}", runner_name);
                        info!("User requested stop for runner: {}", runner_name);
                        if let Err(e) = runner_manager.stop_runner(runner_name).await {
                            *last_error = Some(format!("Failed to stop runner: {}", e));
                        }
                    } else {
                        *last_error = Some("No runner selected. Please select a runner first.".to_string());
                    }
                }
                KeyCode::Left | KeyCode::Right => {
                    let runner_names = runner_manager.get_runner_names();
                    if runner_names.is_empty() { return; }
                    
                    let current_selected = selected.clone();
                    let new_selected_index = if let Some(current_name) = current_selected {
                        runner_names.iter().position(|r| r == &current_name).map(|i| {
                            if key_event.code == KeyCode::Right {
                                (i + 1) % runner_names.len()
                            } else {
                                (i + runner_names.len() - 1) % runner_names.len()
                            }
                        }).unwrap_or(0)
                    } else {
                        0
                    };
                    *selected = Some(runner_names[new_selected_index].clone());
                }
                _ => {}
            }
        });
    }
}

async fn update_status_loop(shared_state: Arc<SharedAppState>, runner_manager: Arc<RunnerManager>) {
    loop {
        match runner_manager.get_runner_status().await {
            Ok(status) => {
                *shared_state.runner_status.write().await = status;
            }
            Err(e) => {
                *shared_state.last_error.lock().await = Some(format!("Failed to update runner status: {}", e));
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}

fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    
    let result = tokio_rt.block_on(async {
        info!("Starting application...");

        // Fix: Properly handle Result before wrapping in Arc
        let config_path = PathBuf::from("config.json");
        let config_manager = Arc::new(ConfigManager::new(config_path).await?);

        let runner_manager = Arc::new(RunnerManager::new(config_manager.clone(), "logs".into()).await?);
        runner_manager.init_models_for_runners().await;
        
        let shared_state = Arc::new(SharedAppState {
            selected_runner: Mutex::new(runner_manager.get_runner_names().first().cloned()),
            ..Default::default()
        });

        // Spawn background tasks
        tokio::spawn(update_status_loop(Arc::clone(&shared_state), runner_manager.clone()));
        tokio::spawn({
            let runner_manager = runner_manager.clone();
            async move {
                if let Err(e) = runner_manager.auto_start_default_runners().await {
                    error!("Failed to auto-start runners: {}", e);
                }
            }
        });

        tokio::spawn({
            let config_manager = config_manager.clone();
            let runner_manager = runner_manager.clone();
            async move {
                if let Err(e) = api::start_api_server(config_manager, runner_manager).await {
                    error!("API server failed to start: {}", e);
                }
            }
        });
        
        let app = Rc::new(App { shared_state: Arc::clone(&shared_state), runner_manager: runner_manager.clone() });
        
        let backend = DomBackend::new().map_err(|e| anyhow::anyhow!("{}", e))?;
        let terminal = Terminal::new(backend).map_err(|e| anyhow::anyhow!("{}", e))?;

        let event_state = Rc::clone(&app);
        terminal.on_key_event(move |key_event| {
            event_state.handle_events(key_event);
        });

        let render_state = Rc::clone(&app);
        terminal.draw_web(move |frame| {
            render_state.render(frame);
        });

        Ok(())
    });
    
    result
}