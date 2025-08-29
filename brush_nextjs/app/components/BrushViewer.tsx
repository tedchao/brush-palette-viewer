'use client';

import { useEffect, useRef, useState } from 'react';
import { CameraSettings, EmbeddedApp, UiMode } from '../../pkg/brush_app';

interface BrushViewerProps {
  url?: string | null;
  fullsplat?: boolean;
  focusDistance?: number;
  minFocusDistance?: number;
  maxFocusDistance?: number;
  speedScale?: number;
}

export default function BrushViewer(props: BrushViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [app, setApp] = useState<EmbeddedApp | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvasId = `brush-canvas-${Date.now()}`;
    canvasRef.current.id = canvasId;

    try {
      const brushApp = new EmbeddedApp(canvasId);
      setApp(brushApp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  useEffect(() => {
    if (app && props.url) {
      app.load_url(props.url);
    }
  }, [app, props.url]);

  useEffect(() => {
    if (app && props.fullsplat) {
      app.set_ui_mode(props.fullsplat ? UiMode.FullScreenSplat : UiMode.Default);
    }
  }, [app, props.url, props.fullsplat]);

  useEffect(() => {
    if (app) {
      app.set_cam_settings(new CameraSettings(
        undefined, // background,
        props.speedScale,
        props.minFocusDistance,
        props.maxFocusDistance,
        undefined, // min_pitch,
        undefined, // max_pitch
        undefined, // min_yaw,
        undefined, // max_yaw
        undefined, // splat_scale
      ));
    }
  }, [app, props.url, props.speedScale, props.minFocusDistance, props.maxFocusDistance]);

  useEffect(() => {
    if (app && props.focusDistance) {
      app.set_cam_focus_distance(props.focusDistance);
    }
  }, [app, props.url, props.focusDistance]);

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      {error ? (
        <div style={{ color: '#ff6b6b' }}>Error: {error}</div>
      ) : (
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block'
          }}
        />
      )}
    </div>
  );
}
