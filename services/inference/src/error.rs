use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct HttpError {
    pub status: u16,
    pub body: ErrorBody,
}

impl HttpError {
    pub fn bad_request(code: &str, message: &str) -> Self {
        Self {
            status: 400,
            body: ErrorBody {
                code: code.to_string(),
                message: message.to_string(),
            },
        }
    }
    pub fn payload_too_large(code: &str, message: &str) -> Self {
        Self {
            status: 413,
            body: ErrorBody {
                code: code.to_string(),
                message: message.to_string(),
            },
        }
    }
    pub fn service_unavailable(code: &str, message: &str) -> Self {
        Self {
            status: 503,
            body: ErrorBody {
                code: code.to_string(),
                message: message.to_string(),
            },
        }
    }
    pub fn internal(code: &str, message: &str) -> Self {
        Self {
            status: 500,
            body: ErrorBody {
                code: code.to_string(),
                message: message.to_string(),
            },
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("invalid output shape")]
    InvalidOutputShape,
    #[error("{0}")]
    Other(String),
}

impl ModelError {
    pub fn code(&self) -> &'static str {
        match self {
            ModelError::InvalidOutputShape => "invalid_output_shape",
            ModelError::Other(_) => "model_error",
        }
    }
}
