# WebRTC Video Pipeline: Frontend Crop → Backend Decode & CV Analysis

## Overview

Replace the current raw-RGBA upload approach with a proper WebRTC ingest pipeline.
The WebRTC video stream is **compressed/encoded**, so the backend cannot receive raw RGBA pixels directly over it.
Instead the backend must accept the encoded video stream, decode it to frames, and feed those frames into the existing CV pipeline.

---

## 1 – Frontend: Crop Center 70% of Decoded Video Frame

- After WebRTC decodes each video frame locally, crop the **center 70%** of the frame (both horizontally and vertically) before sending.
- No face/landmark tracking is required for the crop; a fixed center crop is sufficient.
- The cropped frame (not the full frame) is what gets transported to the backend.

**Acceptance criteria:**
- [ ] Center-crop logic is applied to each decoded video frame on the client side.
- [ ] Crop dimensions are calculated as 70% of the decoded frame's width and height, centered.
- [ ] No tracking library dependency is introduced.

---

## 2 – Transport: Send Cropped Frames via WebRTC Ingest

- The cropped frames are forwarded to the backend as a WebRTC video track (i.e., the backend acts as a WebRTC peer / ingest endpoint, not an HTTP file receiver).
- Each frame sent over the track carries a **timestamp** and is associated with a **session ID** so the backend can correlate frames to the correct analysis session.

**Acceptance criteria:**
- [ ] Frontend establishes a WebRTC peer connection to the backend ingest endpoint.
- [ ] Cropped video frames are sent as a WebRTC video track (encoded).
- [ ] Session ID is signalled during the WebRTC handshake (e.g., as a query parameter on the signalling endpoint or in the SDP offer).

---

## 3 – Backend: Accept WebRTC Stream, Decode Frames, Feed CV Pipeline

- Backend exposes a WebRTC signalling endpoint (offer/answer SDP exchange).
- Once the peer connection is established, the backend receives the encoded video track, **decodes it to raw frames** (e.g., using `aiortc` or equivalent), and passes each decoded frame into the existing `cv_service.py` analysis pipeline.
- **Timestamps** from the video track are preserved and included in analysis results.
- **Session ID** extracted from the signalling request is used to maintain per-session baseline state (reusing the existing session management in `cv_service.py`).

**Acceptance criteria:**
- [ ] Backend accepts a WebRTC offer and returns a valid SDP answer.
- [ ] Backend decodes incoming encoded video frames to numpy arrays (or equivalent) compatible with the existing CV pipeline input.
- [ ] Each decoded frame is passed to the existing `cv_service.py` analysis pipeline without modification to that pipeline's interface.
- [ ] Frame timestamps are recorded and surfaced in analysis results.
- [ ] Session ID is mapped to the per-session baseline state; `DELETE /cv/session` continues to work for cleanup.
- [ ] Backend handles connection teardown and cleans up resources when the WebRTC peer disconnects.

---

## Out of Scope

- Compatibility/migration shims or feature-flag/fallback paths are **not** included in this implementation.
- No changes to the poker inference endpoints (`/poker`).
- No changes to the RL training pipeline.

---

## Labels

`enhancement`
