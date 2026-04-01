"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type ConnectionStatus = "connected" | "disconnected" | "connecting" | "reconnecting";

interface UseWebSocketOptions {
  url: string;
  onMessage: (data: unknown) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface UseWebSocketReturn {
  status: ConnectionStatus;
  sendText: (payload: string) => void;
  sendBinary: (data: ArrayBuffer | Blob) => void;
  connect: () => void;
  disconnect: () => void;
}

export function useWebSocket({
  url,
  onMessage,
  autoReconnect = true,
  reconnectInterval = 2000,
  maxReconnectAttempts = 10,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const cleanup = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    cleanup();
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setStatus("connected");
      attemptRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessageRef.current(data);
      } catch {
        onMessageRef.current(event.data);
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      if (autoReconnect && attemptRef.current < maxReconnectAttempts) {
        const delay = Math.min(reconnectInterval * Math.pow(1.5, attemptRef.current), 30000);
        attemptRef.current++;
        setStatus("reconnecting");
        reconnectTimerRef.current = setTimeout(connect, delay);
      }
    };

    ws.onerror = () => {};

    wsRef.current = ws;
  }, [url, autoReconnect, reconnectInterval, maxReconnectAttempts, cleanup]);

  const disconnect = useCallback(() => {
    cleanup();
    attemptRef.current = maxReconnectAttempts; // prevent reconnect
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("disconnected");
  }, [cleanup, maxReconnectAttempts]);

  const sendText = useCallback((payload: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(payload);
    }
  }, []);

  const sendBinary = useCallback((data: ArrayBuffer | Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  useEffect(() => {
    return () => {
      cleanup();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [cleanup]);

  return { status, sendText, sendBinary, connect, disconnect };
}
