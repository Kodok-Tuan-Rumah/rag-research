use crate::error::HttpError;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;
use std::thread;
use std::time::{Duration, Instant};

pub trait BatchModel<Req, Res> {
    fn infer(&mut self, batch: &[Req]) -> Result<Vec<Res>, String>;
}

pub struct WorkerPool<M, Req, Res>
where
    M: BatchModel<Req, Res> + Send + 'static,
    Req: Send + 'static,
    Res: Send + 'static + Clone,
{
    tx: mpsc::SyncSender<Job<Req, Res>>,
    _marker: PhantomData<M>,
}

struct Job<Req, Res> {
    items: Vec<Req>,
    reply: mpsc::Sender<Result<Vec<Res>, String>>,
}

impl<M, Req, Res> WorkerPool<M, Req, Res>
where
    M: BatchModel<Req, Res> + Send + 'static,
    Req: Send + 'static,
    Res: Send + 'static + Clone,
{
    pub fn new(
        model: M,
        workers: usize,
        capacity: usize,
        batch_window_ms: u64,
        max_batch_size: usize,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel::<Job<Req, Res>>(capacity);
        let rx = Arc::new(Mutex::new(rx));
        let model = Arc::new(Mutex::new(model));
        let n = workers.max(1);
        for _ in 0..n {
            let rx = Arc::clone(&rx);
            let model = Arc::clone(&model);
            let window = Duration::from_millis(batch_window_ms);
            let max_bs = max_batch_size;
            thread::spawn(move || {
                loop {
                    // wait for first job
                    let first = {
                        let lock = rx.lock().unwrap();
                        lock.recv()
                    };
                    let first = match first {
                        Ok(j) => j,
                        Err(_) => break,
                    };
                    let mut counts: Vec<usize> = vec![first.items.len()];
                    let mut all_items: Vec<Req> = first.items;
                    let mut replies: Vec<mpsc::Sender<Result<Vec<Res>, String>>> = vec![first.reply];
                    let start = Instant::now();
                    while all_items.len() < max_bs {
                        let remaining = window.saturating_sub(start.elapsed());
                        if remaining.is_zero() {
                            break;
                        }
                        let next = {
                            let lock = rx.lock().unwrap();
                            lock.recv_timeout(remaining)
                        };
                        match next {
                            Ok(job) => {
                                counts.push(job.items.len());
                                all_items.extend(job.items);
                                replies.push(job.reply);
                            }
                            Err(_) => break,
                        }
                    }
                    // run inference
                    let mut m = model.lock().unwrap();
                    let res = m.infer(&all_items);
                    match res {
                        Ok(all) => {
                            // split
                            let mut cursor = 0usize;
                            for (i, reply) in replies.into_iter().enumerate() {
                                let n = counts[i];
                                let chunk = all[cursor..cursor + n].to_vec();
                                cursor += n;
                                let _ = reply.send(Ok(chunk));
                            }
                        }
                        Err(e) => {
                            for reply in replies {
                                let _ = reply.send(Err(e.clone()));
                            }
                        }
                    }
                }
            });
        }
        Self {
            tx,
            _marker: PhantomData,
        }
    }

    pub fn submit(&self, items: Vec<Req>) -> Result<Vec<Res>, HttpError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        let job = Job {
            items,
            reply: reply_tx,
        };
        self.tx
            .send(job)
            .map_err(|_| HttpError::service_unavailable("queue_full", "inference queue is full"))?;
        reply_rx
            .recv()
            .map_err(|_| HttpError::service_unavailable("queue_full", "worker dropped"))?
            .map_err(|e| HttpError::internal("model_error", &e))
    }
}
