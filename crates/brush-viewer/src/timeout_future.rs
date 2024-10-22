use gloo_timers::future::TimeoutFuture;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use web_time::{Duration, Instant};

/// Wrap a future to yield back to the browser macrotasks
/// queue each time it yields (using a setTimeout).
///
/// In the future this could work better with scheduler.yield.
pub fn with_timeout_yield<F: Future>(
    future: F,
    min_time_between_yields: Duration,
) -> impl Future<Output = F::Output> {
    struct WithTimeoutYield<F> {
        future: Pin<Box<F>>,
        timeout: Option<TimeoutFuture>,
        last_yield: Option<Instant>,
        min_duration: Duration,
    }

    impl<F: Future> Future for WithTimeoutYield<F> {
        type Output = F::Output;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let this = self.get_mut();

            if let Some(timeout) = &mut this.timeout {
                match std::pin::pin!(timeout).poll(cx) {
                    Poll::Ready(_) => {
                        this.timeout = None;
                        this.last_yield = Some(Instant::now());
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }

            match this.future.as_mut().poll(cx) {
                Poll::Ready(output) => Poll::Ready(output),
                Poll::Pending => {
                    let should_yield = this
                        .last_yield
                        .map(|t| t.elapsed() >= this.min_duration)
                        .unwrap_or(true);

                    if should_yield && this.timeout.is_none() {
                        this.timeout = Some(TimeoutFuture::new(0));
                    }
                    Poll::Pending
                }
            }
        }
    }

    WithTimeoutYield {
        future: Box::pin(future),
        timeout: None,
        last_yield: None,
        min_duration: min_time_between_yields,
    }
}
