use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use crate::models::{WsGreekItem, WsIndexTick};

/// In-memory cache for latest ticks — shared across threads
#[derive(Default, Clone)]
pub struct TickCache {
    pub greeks: Arc<RwLock<HashMap<i64, WsGreekItem>>>,   // inst_id -> latest greek tick
    pub indexes: Arc<RwLock<HashMap<String, WsIndexTick>>>, // indexname -> latest tick
    pub session_token: Arc<RwLock<Option<String>>>,
}

impl TickCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_greek(&self, item: WsGreekItem) {
        if let Ok(mut map) = self.greeks.write() {
            map.insert(item.inst_id, item);
        }
    }

    pub fn update_index(&self, item: WsIndexTick) {
        if let Ok(mut map) = self.indexes.write() {
            map.insert(item.indexname.clone(), item);
        }
    }

    pub fn get_all_greeks(&self) -> Vec<WsGreekItem> {
        self.greeks.read().map(|m| m.values().cloned().collect()).unwrap_or_default()
    }

    pub fn get_all_indexes(&self) -> Vec<WsIndexTick> {
        self.indexes.read().map(|m| m.values().cloned().collect()).unwrap_or_default()
    }

    pub fn set_token(&self, token: String) {
        if let Ok(mut t) = self.session_token.write() {
            *t = Some(token);
        }
    }

    pub fn get_token(&self) -> Option<String> {
        self.session_token.read().ok()?.clone()
    }
}
