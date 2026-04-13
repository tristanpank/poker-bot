"use client";

function stripTrailingSlash(value: string): string {
  return value.replace(/\/$/, "");
}

export function getBackendBaseUrl(): string {
  const explicitUrl = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();
  if (explicitUrl) {
    return stripTrailingSlash(explicitUrl);
  }

  if (typeof window !== "undefined") {
    const { hostname, protocol } = window.location;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      return "http://localhost:8000";
    }

    return `${protocol}//${hostname}:8000`;
  }

  return "http://localhost:8000";
}
