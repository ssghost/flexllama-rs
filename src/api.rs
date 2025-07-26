// src/api.rs

use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};
use axum::{
    body::{Body, Bytes},
    extract::{Path, State},
    http::{
        header::{ACCEPT, CONTENT_TYPE},
        Method, Response, StatusCode,
    },
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::fs;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, debug, warn};

use crate::{
    config::ConfigManager,
    runner::{HealthMessages, HealthStatus, RunnerManager},
}; // Assuming config.rs and runner.rs are in the same crate

/// The main API server structure.
pub struct APIServer {
    config_manager: Arc<ConfigManager>,
    runner_manager: Arc<RunnerManager>,
    host: String,
    port: u16,
}

impl APIServer {
    pub fn new(config_manager: Arc<ConfigManager>, runner_manager: Arc<RunnerManager>) -> Self {
        let host = config_manager.get_api_host().to_string();
        let port = config_manager.get_api_port();
        Self {
            config_manager,
            runner_manager,
            host,
            port,
        }
    }

    pub async fn start(&self) -> Result<(), anyhow::Error> {
        info!("Starting API server on {}:{}", self.host, self.port);

        let app_state = Arc::new(AppState {
            config_manager: self.config_manager.clone(),
            runner_manager: self.runner_manager.clone(),
            frontend_dir: PathBuf::from("frontend"), // Path to the frontend static files
        });

        // CORS layer, similar to Python's handle_options
        let cors = CorsLayer::new()
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(vec![CONTENT_TYPE, ACCEPT])
            .allow_origin(Any);

        let app = Router::new()
            // Dashboard routes
            .route("/", get(handle_dashboard))
            .route("/dashboard", get(handle_dashboard))
            // Static files for frontend
            // In a production setup, serve these with a dedicated web server (nginx, apache)
            // or use a more robust static file serving crate like `tower-http::services::ServeDir`
            // For now, let's serve them dynamically or assume they are served externally.
            // Axum's ServeDir requires a path, so let's simplify for the example.
            // The original Python uses `web.static("/frontend", "frontend", show_index=True)`.
            // For a direct Rust equivalent, we'll serve files under /frontend explicitly.
            .route("/frontend/:file_name", get(serve_frontend_file))
            .route("/frontend/:dir/:file_name", get(serve_frontend_nested_file))


            // OpenAI-compatible API routes
            .route("/v1/models", get(handle_models))
            .route("/health", get(handle_health))
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/v1/completions", post(handle_completions))
            .route("/v1/embeddings", post(handle_embeddings))
            .route("/v1/rerank", post(handle_rerank))
            .with_state(app_state.clone())
            .layer(cors);

        let addr: SocketAddr = format!("{}:{}", self.host, self.port).parse()?;
        info!("API server listening on {}", addr);

        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await?;

        Ok(())
    }

    pub async fn stop(&self) -> Result<(), anyhow::Error> {
        info!("API server stopping...");
        // Axum server runs in its own task, dropping the server handle will stop it.
        // For a more graceful shutdown, you might use a `CancellationToken` or similar.
        info!("API server stopped.");
        Ok(())
    }

    pub fn get_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

// Application State shared across handlers
#[derive(Clone)]
struct AppState {
    config_manager: Arc<ConfigManager>,
    runner_manager: Arc<RunnerManager>,
    frontend_dir: PathBuf,
}

// --- Handler Functions ---

async fn handle_dashboard(State(state): State<Arc<AppState>>) -> Result<Response<Body>, StatusCode> {
    let dashboard_path = state.frontend_dir.join("index.html");
    if !dashboard_path.exists() {
        error!("Dashboard file not found: {:?}", dashboard_path);
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Dashboard not found."))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
    }

    match fs::read_to_string(&dashboard_path).await {
        Ok(content) => Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "text/html")
            .body(Body::from(content))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR),
        Err(e) => {
            error!("Error reading dashboard file {:?}: {}", dashboard_path, e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("Error loading dashboard: {}", e)))
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// Generic handler for serving frontend static files
async fn serve_frontend_file(
    State(state): State<Arc<AppState>>,
    Path(file_name): Path<String>,
) -> Result<Response<Body>, StatusCode> {
    let file_path = state.frontend_dir.join(&file_name);
    serve_static_file(file_path).await
}

async fn serve_frontend_nested_file(
    State(state): State<Arc<AppState>>,
    Path((dir, file_name)): Path<(String, String)>,
) -> Result<Response<Body>, StatusCode> {
    let file_path = state.frontend_dir.join(&dir).join(&file_name);
    serve_static_file(file_path).await
}

async fn serve_static_file(path: PathBuf) -> Result<Response<Body>, StatusCode> {
    if !path.exists() || !path.is_file() {
        debug!("Static file not found: {:?}", path);
        return Err(StatusCode::NOT_FOUND);
    }

    let mime_type = mime_guess::from_path(&path)
        .first_or_octet_stream()
        .to_string();

    match fs::read(&path).await {
        Ok(content) => Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, mime_type)
            .body(Body::from(content))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR),
        Err(e) => {
            error!("Error reading static file {:?}: {}", path, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}


async fn handle_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut models = Vec::new();
    let model_aliases = state.config_manager.get_model_aliases();

    for alias in model_aliases {
        models.push(json!({
            "id": alias,
            "object": "model",
            "created": chrono::Local::now().timestamp(), // Unix timestamp
            "owned_by": "user",
        }));
    }

    Json(json!({
        "object": "list",
        "data": models
    }))
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    // Check which runners are active and their current models
    let mut active_runners_map = HashMap::new();
    let mut runner_current_models_map = HashMap::new();
    let mut runner_info_map = HashMap::new();

    for runner_name in state.runner_manager.get_runner_names() {
        let is_active = state.runner_manager.is_runner_running(&runner_name).await.unwrap_or(false);
        active_runners_map.insert(runner_name.clone(), is_active);

        let current_model = state.runner_manager.get_current_model_for_runner(&runner_name).await;
        runner_current_models_map.insert(runner_name.clone(), current_model.clone());

        if let Some(runner_cfg) = state.config_manager.get_runner_config(&runner_name) {
            runner_info_map.insert(runner_name.clone(), json!({
                "host": runner_cfg.host.clone().unwrap_or_else(|| state.config_manager.get_api_host().to_string()),
                "port": runner_cfg.port, // Port might be None if not explicitly set
                "current_model": current_model,
                "is_active": is_active,
            }));
        }
    }

    // Check actual model health from llama.cpp health endpoints (via RunnerManager)
    let mut model_health_map = HashMap::new();
    for model_alias in state.config_manager.get_model_aliases() {
        let health_data = state.runner_manager.check_model_health(&model_alias).await;
        model_health_map.insert(model_alias, health_data);
    }

    Json(json!({
        "status": HealthStatus::Ok.as_str(),
        "active_runners": active_runners_map,
        "runner_current_models": runner_current_models_map,
        "runner_info": runner_info_map,
        "model_health": model_health_map,
    }))
}

// Unified request handler for /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/rerank
async fn handle_forwarded_request(
    State(state): State<Arc<AppState>>,
    Path(endpoint_tail): Path<String>, // e.g., "chat/completions"
    Json(mut payload): Json<Value>,
) -> Result<Response<Body>, StatusCode> {
    let full_endpoint = format!("/v1/{}", endpoint_tail);
    debug!("Received request for {}", full_endpoint);

    let model_alias = extract_model_alias(&payload, &state.config_manager).await;

    let model_alias = match model_alias {
        Some(alias) => alias,
        None => {
            error!("Model not specified in request payload or no default available.");
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Pre-flight readiness check with retry
    let (is_ready, error_message) = state.runner_manager.ensure_model_ready_with_retry(&model_alias).await;
    if !is_ready {
        error!("Model {} not ready: {:?}", model_alias, error_message);
        return Ok(Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .json(&json!({
                "error": {
                    "message": format!("Model not ready: {}", error_message.unwrap_or_else(|| "Unknown error".to_string())),
                    "type": "model_not_ready",
                }
            }))
            .unwrap());
    }

    let is_streaming = payload["stream"].as_bool().unwrap_or(false);

    if is_streaming {
        // Forward streaming request
        let runner = state.runner_manager.get_runner_for_model(&model_alias)
            .ok_or_else(|| {
                error!("Runner not found for model: {}", model_alias);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        
        let target_url = format!("http://{}{}", runner.port, full_endpoint); // Use runner.host and runner.port

        debug!("Forwarding streaming request to {}", target_url);

        let client = reqwest::Client::new();
        let request_builder = client.post(&target_url).json(&payload);

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Error sending streaming request to {}: {}", target_url, e);
                return Err(StatusCode::SERVICE_UNAVAILABLE);
            }
        };

        let status = response.status();
        let headers = response.headers().clone();

        if !status.is_success() {
            let error_body = response.bytes().await.unwrap_or_default();
            error!("Streaming request failed with status {}: {:?}", status, String::from_utf8_lossy(&error_body));
            return Ok(Response::builder()
                .status(status)
                .body(Body::from(error_body))
                .unwrap());
        }

        let mut axum_response_builder = Response::builder()
            .status(status)
            .header(CONTENT_TYPE, headers.get(CONTENT_TYPE).cloned().unwrap_or_else(|| "text/event-stream".parse().unwrap()))
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive");

        // Stream the response body directly
        let stream = response.bytes_stream();
        let body = Body::from_stream(stream);

        axum_response_builder.body(body).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)

    } else {
        // Forward non-streaming request
        let (success, response_data, status_code) = state.runner_manager.forward_request(&model_alias, &full_endpoint, payload, 120).await;

        Ok(Response::builder()
            .status(status_code)
            .json(&response_data)
            .unwrap())
    }
}

// Wrapper handlers for specific OpenAI-compatible endpoints
async fn handle_chat_completions(
    state: State<Arc<AppState>>,
    payload: Json<Value>,
) -> Result<Response<Body>, StatusCode> {
    handle_forwarded_request(state, Path("chat/completions".to_string()), payload).await
}

async fn handle_completions(
    state: State<Arc<AppState>>,
    payload: Json<Value>,
) -> Result<Response<Body>, StatusCode> {
    handle_forwarded_request(state, Path("completions".to_string()), payload).await
}

async fn handle_embeddings(
    state: State<Arc<AppState>>,
    payload: Json<Value>,
) -> Result<Response<Body>, StatusCode> {
    handle_forwarded_request(state, Path("embeddings".to_string()), payload).await
}

async fn handle_rerank(
    state: State<Arc<AppState>>,
    payload: Json<Value>,
) -> Result<Response<Body>, StatusCode> {
    handle_forwarded_request(state, Path("rerank".to_string()), payload).await
}


// Helper to extract model alias from request payload
async fn extract_model_alias(payload: &Value, config_manager: &ConfigManager) -> Option<String> {
    if let Some(model_value) = payload.get("model") {
        if let Some(model_str) = model_value.as_str() {
            return Some(model_str.to_string());
        }
    }
    
    // Default to first model if not specified, similar to Python
    config_manager.get_model_aliases().first().cloned()
}

// Trait and impl to allow .json() on ResponseBuilder
trait ResponseBuilderExt {
    fn json(self, json_data: &Value) -> Result<Response<Body>, http::Error>;
}

impl ResponseBuilderExt for http::response::Builder {
    fn json(self, json_data: &Value) -> Result<Response<Body>, http::Error> {
        self.header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(json_data).unwrap_or_default()))
    }
}