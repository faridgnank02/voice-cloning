(() => {
  const state = { socket: null, recorder: null, stream: null, audioContext: null, processor: null, audio: [], playing: false };
  const $ = (id) => document.getElementById(id);
  const status = (text) => { $("status").textContent = text; };
  const send = (value) => state.socket && state.socket.readyState === WebSocket.OPEN && state.socket.send(value);

  function playAudio(base64) {
    const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
    state.audio.push(new Blob([bytes], { type: "audio/wav" }));
    if (!state.playing) playNext();
  }

  function playNext() {
    const blob = state.audio.shift();
    if (!blob) { state.playing = false; return; }
    state.playing = true;
    const audio = new Audio(URL.createObjectURL(blob));
    audio.onended = () => { URL.revokeObjectURL(audio.src); playNext(); };
    audio.play().catch(() => { state.playing = false; });
  }

  function handleEvent(event) {
    if (event.type === "session_started") {
      status("Connected"); $("record").disabled = false; $("interrupt").disabled = false; return;
    }
    if (event.type === "transcript_delta") $("transcript").textContent += event.text || "";
    if (event.type === "assistant_text_delta") $("assistant").textContent += event.text || "";
    if (event.type === "response_audio" && event.audio?.data) playAudio(event.audio.data);
    if (event.type === "turn_completed") status("Ready");
    if (event.type === "turn_interrupted") { state.audio = []; status("Interrupted"); }
    if (event.type === "error") status(`Error: ${event.message || "request failed"}`);
  }

  $("connect").onclick = () => {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    state.socket = new WebSocket(`${scheme}://${location.host}/ws`);
    state.socket.onopen = () => {
      status("Authorizing…");
      send(JSON.stringify({ type: "start_session", voice_profile_id: $("profile-id").value, language: $("language").value }));
    };
    state.socket.onmessage = async (message) => {
      if (typeof message.data === "string") handleEvent(JSON.parse(message.data));
    };
    state.socket.onclose = () => { status("Disconnected"); $("record").disabled = true; };
  };

  $("record").onclick = async () => {
    if (state.processor) {
      state.processor.disconnect(); state.processor = null;
      state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null;
      await state.audioContext?.close(); state.audioContext = null;
      $("record").textContent = "Start recording"; send(JSON.stringify({ type: "finish_turn" })); return;
    }
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    state.audioContext = new AudioContext({ sampleRate: 16000 });
    const source = state.audioContext.createMediaStreamSource(state.stream);
    state.processor = state.audioContext.createScriptProcessor(2048, 1, 1);
    state.processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) pcm[i] = Math.max(-1, Math.min(1, input[i])) * 0x7fff;
      send(pcm.buffer);
    };
    source.connect(state.processor); state.processor.connect(state.audioContext.destination);
    $("transcript").textContent = ""; $("assistant").textContent = ""; $("record").textContent = "Stop recording"; status("Listening…");
  };

  $("interrupt").onclick = () => { state.audio = []; send(JSON.stringify({ type: "interrupt" })); };
})();
