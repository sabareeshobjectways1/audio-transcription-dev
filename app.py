import streamlit as st
import json
import uuid
import base64
from datetime import datetime
import requests
import io
from pydub import AudioSegment

# --- Page Configuration ---
st.set_page_config(
    page_title="Audio Annotation Tool",
    page_icon="🎙️",
    layout="wide"
)

# --- App Constants ---
# Files larger than this will be pre-processed for the player.
LARGE_FILE_THRESHOLD_MB = 25

# --- Initialize Session State ---
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'speakers' not in st.session_state:
    st.session_state.speakers = []
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'page_state' not in st.session_state:
    st.session_state.page_state = 'metadata_input'
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None
if 'transcription_content' not in st.session_state:
    st.session_state.transcription_content = ""

# --- Helper Functions ---

def get_json_download_link(data, filename="annotated_data.json"):
    """Generates a link to download the annotated JSON data."""
    json_str = json.dumps(data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}">Download JSON File</a>'

def process_audio_for_player(audio_bytes: bytes):
    """
    Optimizes large audio files for the wavesurfer.js player to prevent crashes.
    Returns the processed audio bytes and the format.
    """
    file_size_mb = len(audio_bytes) / (1024 * 1024)
    
    if file_size_mb <= LARGE_FILE_THRESHOLD_MB:
        # For small files, just convert to WAV for consistency
        st.info("Small audio file detected. Using original quality for player.")
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            return buf.getvalue(), "wav"
        except Exception:
             # If it's already a format pydub handles, just pass it through, browser will try to decode
             return audio_bytes, "wav" # Assume wav, most browsers can handle it.
    else:
        st.warning(f"Large file detected ({file_size_mb:.1f} MB). Optimizing for player performance...")
        with st.spinner("Creating a lightweight audio preview... (This may take a moment)"):
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                
                # --- Optimization Steps ---
                # 1. Convert to mono
                audio = audio.set_channels(1)
                # 2. Downsample for speech (22.05 kHz is sufficient)
                audio = audio.set_frame_rate(22050)
                
                # 3. Export as a compressed format (MP3) with a lower bitrate
                buf = io.BytesIO()
                audio.export(buf, format="mp3", bitrate="64k")
                
                st.success("Optimization complete! Player is now ready.")
                return buf.getvalue(), "mp3"
            except Exception as e:
                st.error(f"Failed to optimize audio file. Error: {e}")
                return None, None


def transcribe_audio_segment_with_gemini(full_audio_bytes, start_time, end_time, api_key):
    """
    Extracts the audio segment from the ORIGINAL high-quality audio bytes.
    """
    try:
        # ALWAYS use the original full_audio_bytes for transcription
        audio = AudioSegment.from_file(io.BytesIO(full_audio_bytes))
        
        start_ms = int(float(start_time) * 1000)
        end_ms = int(float(end_time) * 1000)
        segment = audio[start_ms:end_ms]

        buf = io.BytesIO()
        segment.export(buf, format="wav")
        segment_bytes = buf.getvalue()
        audio_base64 = base64.b64encode(segment_bytes).decode()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}

        duration = end_time - start_time
        prompt = f"""You are an expert audio transcriptionist... (Your prompt here)""" # Shortened for brevity

        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "audio/wav", "data": audio_base64}}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000, "topP": 0.8, "topK": 40}
        }

        with st.spinner(f"Requesting transcription for segment {start_time}s - {end_time}s..."):
            response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            parts = result.get('candidates', [{}])[0].get('content', {}).get('parts', [])
            if parts:
                return parts[0].get('text', '').strip()
            return "[NO_CONTENT]"
        else:
            st.error(f"Gemini API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# =====================================================================================
# CUSTOM AUDIO PLAYER COMPONENT (UPDATED)
# =====================================================================================

def audio_player_component(audio_bytes: bytes, audio_format: str = "wav"):
    """
    Creates a custom audio player component using wavesurfer.js.
    Now accepts an audio_format to build the correct data URI.
    """
    b64_audio = base64.b64encode(audio_bytes).decode()
    component_html = f"""
    <div id="waveform-container" style="border: 1px solid #ddd; border-radius: 8px; padding: 20px; width: 90%;">
        <div id="waveform"></div>
        <div style="margin-top: 15px; display: flex; align-items: center; gap: 20px;">
            <button id="playBtn" style="padding: 8px 16px; border-radius: 5px; border: 1px solid #ccc; cursor: pointer;">Play</button>
            <div style="font-family: monospace; font-size: 1.2em;">
                Current Time: <span id="time-display">0.000</span> s
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <label for="playbackSpeed">Speed:</label>
                <select id="playbackSpeed" style="border-radius: 5px; padding: 5px;">
                    <option value="0.5">0.5x</option><option value="1" selected>1.0x</option>
                    <option value="1.5">1.5x</option><option value="2">2.0x</option>
                </select>
            </div>
        </div>
    </div>
    <script src="https://unpkg.com/wavesurfer.js"></script>
    <script>
        var wavesurfer = WaveSurfer.create({{
            container: '#waveform', waveColor: 'violet', progressColor: 'purple',
            barWidth: 2, barRadius: 3, height: 100, barGap: 3,
            responsive: true, fillParent: true, minPxPerSec: 1,
            cursorWidth: 1, cursorColor: 'purple'
        }});
        // Use the dynamic audio_format here
        wavesurfer.load('data:audio/{audio_format};base64,{b64_audio}');
        const playBtn = document.getElementById('playBtn');
        const timeDisplay = document.getElementById('time-display');
        const speedSelector = document.getElementById('playbackSpeed');
        playBtn.onclick = function() {{ wavesurfer.playPause(); }};
        wavesurfer.on('audioprocess', function() {{ timeDisplay.textContent = wavesurfer.getCurrentTime().toFixed(3); }});
        wavesurfer.on('interaction', function() {{ timeDisplay.textContent = wavesurfer.getCurrentTime().toFixed(3); }});
        speedSelector.onchange = function() {{ wavesurfer.setPlaybackRate(this.value); }};
        wavesurfer.on('finish', function () {{ playBtn.textContent = 'Play'; }});
        wavesurfer.on('play', function () {{ playBtn.textContent = 'Pause'; }});
        wavesurfer.on('pause', function () {{ playBtn.textContent = 'Play'; }});
    </script>
    """
    st.components.v1.html(component_html, height=200)

# =====================================================================================
# PAGE 1: METADATA INPUT FORM (Unchanged)
# =====================================================================================
def metadata_form():
    st.title("Step 1: Input Metadata")
    st.markdown("---")
    # ... (This entire function is unchanged and can be copied from your version) ...
    with st.form(key="metadata_form"):
        st.subheader("1. Type")
        type_name = st.text_input("Name", "MULTI_SPEAKER_LONG_FORM_TRANSCRIPTION")
        type_version = st.text_input("Version", "3.1")
        st.subheader("2. Language")
        lang_full = st.text_input("Full Language Name", "en_NZ")
        lang_short = st.text_input("Short Name / Symbol", "en_NZ")
        st.subheader("3. Person in Audio")
        head_count = st.number_input("Head Count", min_value=1, value=1, step=1)
        st.subheader("4. Domain")
        domain_name = st.text_input("Domain Name", "Call-center")
        topic_list = st.text_input("Topic List (comma-separated)", "Banking")
        st.subheader("5. Annotator Info")
        login_encrypted = st.text_input("Login Encrypted (Optional)", "")
        annotator_id = st.text_input("Annotator ID", "t5fb5aa2")
        st.subheader("6. Convention Info")
        master_convention = st.text_input("Master Convention Name", "awsTranscriptionGuidelines_en_US_3.1")
        custom_addendum = st.text_input("Custom Addendum (Optional)", "en_NZ_1.0")
        st.subheader("7. Speaker Details")
        speakers_input = []
        speaker_dominant_varieties_data = []
        for i in range(int(head_count)):
            st.markdown(f"**Speaker {i+1}**")
            speaker_id = st.text_input(f"Speaker ID (leave blank for auto)", key=f"speaker_id_{i}")
            gender = st.selectbox(f"Gender", ["Female", "Male", "Other"], key=f"gender_{i}")
            gender_source = st.text_input(f"Gender Source", "Annotator", key=f"gender_source_{i}")
            speaker_nativity = st.selectbox(f"Speaker Nativity", ["Native", "Non-Native"], key=f"nativity_{i}")
            speaker_nativity_source = st.text_input(f"Speaker Nativity Source", "Annotator", key=f"nativity_source_{i}")
            speaker_role = st.text_input(f"Speaker Role", "Customer", key=f"role_{i}")
            speaker_role_source = st.text_input(f"Speaker Role Source", "Annotator", key=f"role_source_{i}")
            st.markdown(f"*Speaker Language Info*")
            language_locale = st.text_input(f"Language Locale", lang_short, key=f"lang_locale_{i}")
            language_variety = st.text_input(f"Language Variety (comma-separated)", key=f"lang_variety_{i}")
            other_language_influence = st.text_input(f"Other Language Influence (comma-separated)", key=f"other_lang_influence_{i}")
            speakers_input.append({"speakerId": speaker_id if speaker_id else str(uuid.uuid4()),"gender": gender,"genderSource": gender_source,"speakerNativity": speaker_nativity,"speakerNativitySource": speaker_nativity_source,"speakerRole": speaker_role,"speakerRoleSource": speaker_role_source,"languages": [language_locale]})
            if i == 0:
                 speaker_dominant_varieties_data.append({"languageLocale": language_locale,"languageVariety": [v.strip() for v in language_variety.split(",") if v.strip()],"otherLanguageInfluence": [v.strip() for v in other_language_influence.split(",") if v.strip()]})
        if st.form_submit_button(label="Save Metadata and Proceed to Annotation"):
            st.session_state.metadata = {"type": {"name": type_name, "version": type_version},"languageInfo": {"spokenLanguages": [lang_full], "speakerDominantVarieties": speaker_dominant_varieties_data},"domainInfo": {"domainVersion": "1.0", "domainList": [{"domain": domain_name, "topicList": [t.strip() for t in topic_list.split(',')]}]},"annotatorInfo": {"loginEncrypted": login_encrypted, "annotatorId": annotator_id},"conventionInfo": {"masterConventionName": master_convention, "customAddendum": custom_addendum},"internalLanguageCode": lang_short}
            st.session_state.speakers = speakers_input
            st.session_state.page_state = 'annotation'
            st.success("Metadata saved successfully!")
            st.rerun()

# =====================================================================================
# PAGE 2: AUDIO ANNOTATION (UPDATED WITH PRE-PROCESSING LOGIC)
# =====================================================================================

def annotation_page():
    st.title("Step 2: Audio Annotation")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])

    if uploaded_file:
        if st.session_state.current_audio is None or st.session_state.current_audio.get('name') != uploaded_file.name:
            # Store the original, high-quality audio bytes in session state
            st.session_state.current_audio = {'name': uploaded_file.name, 'bytes': uploaded_file.getvalue()}
            # Process the audio to create a potentially smaller version for the player
            player_bytes, player_format = process_audio_for_player(st.session_state.current_audio['bytes'])
            st.session_state.current_audio['player_bytes'] = player_bytes
            st.session_state.current_audio['player_format'] = player_format

        # Use the original bytes for analysis and properties
        original_audio_bytes = st.session_state.current_audio['bytes']
        
        # Use the (potentially smaller) processed bytes for the player
        player_audio_bytes = st.session_state.current_audio.get('player_bytes')
        player_audio_format = st.session_state.current_audio.get('player_format')

        st.subheader("Audio File Properties (from original file)")
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(original_audio_bytes))
            # ... (property display logic is unchanged) ...
            duration_seconds = len(audio_segment) / 1000.0; peak_loudness_dbfs = audio_segment.max_dBFS; sample_rate_khz = audio_segment.frame_rate / 1000.0; channels = "Stereo" if audio_segment.channels >= 2 else "Mono"
            col1, col2, col3, col4 = st.columns(4); col1.metric(label="Duration", value=f"{duration_seconds:.2f} s"); col2.metric(label="Peak Loudness", value=f"{peak_loudness_dbfs:.2f} dBFS"); col3.metric(label="Sample Rate", value=f"{sample_rate_khz:.1f} kHz"); col4.metric(label="Channels", value=channels)
        except Exception as e:
            st.error(f"Could not read audio properties. Error: {e}")

        st.subheader("Audio Player")
        if player_audio_bytes and player_audio_format:
            # Pass the optimized bytes and format to the player component
            audio_player_component(player_audio_bytes, player_audio_format)
        else:
            st.error("Audio could not be processed for the player.")
        
        # --- The rest of the page uses original_audio_bytes for transcription ---
        st.subheader("Add a New Segment")
        # ... (rest of the annotation logic is unchanged and can be copied from your version) ...
        time_col1, time_col2, transcribe_col = st.columns([2, 2, 1])
        with time_col1: start_time = st.text_input("Start Time (s)", "0.0", key="start_time_input")
        with time_col2: end_time = st.text_input("End Time (s)", "5.0", key="end_time_input")
        with transcribe_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎙️ Transcribe", help="Transcribe this audio segment using Gemini"):
                try:
                    start_float, end_float = float(start_time), float(end_time)
                    if not(start_float < end_float and start_float >= 0): st.error("Start time must be less than end time and not negative.")
                    else:
                        api_key = st.secrets["GEMINI_API_KEY"]
                        # Pass the ORIGINAL bytes for high-quality transcription
                        transcription = transcribe_audio_segment_with_gemini(original_audio_bytes, start_float, end_float, api_key)
                        if transcription is not None:
                            if transcription in ["[SILENCE]", "[NOISE]", "[NO_CONTENT]"]: st.info(f"API response: {transcription}"); st.session_state.transcription_content = transcription if transcription == "[NOISE]" else ""
                            else: st.session_state.transcription_content = transcription; st.success(f"✅ Transcribed segment ({start_float}s - {end_float}s) successfully!")
                            st.rerun()
                except (ValueError, KeyError) as e: st.error(f"Error: {e}")
        
        with st.form(key="segment_form", clear_on_submit=True):
            transcription = st.text_area("Transcription Content", value=st.session_state.transcription_content, help="Use the 'Transcribe' button to auto-fill this field")
            c1, c2, c3 = st.columns(3)
            with c1: primary_type = st.selectbox("Primary Type", ["Speech", "Noise", "Music", "Silence"])
            with c2: loudness_level = st.selectbox("Loudness Level", ["Normal", "Quiet", "Loud"])
            with c3:
                if st.session_state.speakers: speaker_options = {s['speakerId']: f"Speaker {i+1} ({s.get('speakerRole', 'N/A')})" for i, s in enumerate(st.session_state.speakers)}; selected_speaker_id = st.selectbox("Speaker", options=list(speaker_options.keys()), format_func=lambda x: speaker_options[x])
                else: st.warning("No speakers defined."); selected_speaker_id = None
            if st.form_submit_button("Add Segment"):
                if selected_speaker_id:
                    try:
                        start_float, end_float = float(start_time), float(end_time)
                        if start_float >= end_float: st.error("Start time must be less than end time!")
                        else:
                            lang_code = st.session_state.metadata.get('internalLanguageCode', 'en_US')
                            st.session_state.segments.append({"start": start_float,"end": end_float,"segmentId": str(uuid.uuid4()),"primaryType": primary_type,"loudnessLevel": loudness_level,"language": lang_code,"segmentLanguages": [lang_code],"speakerId": selected_speaker_id,"transcriptionData": {"content": transcription}})
                            st.session_state.transcription_content = ""; st.success("Segment added!"); st.rerun()
                    except ValueError: st.error("Invalid start/end times.")
                else: st.error("Cannot add segment without a speaker.")

    if st.session_state.segments:
        # ... (rest of the page is unchanged and can be copied from your version) ...
        st.subheader("Annotated Segments")
        st.session_state.segments = sorted(st.session_state.segments, key=lambda x: x.get('start', 0))
        for i, seg in enumerate(st.session_state.segments):
            with st.expander(f"Segment {i+1}: {seg['start']}s - {seg['end']}s ({seg['primaryType']})"):
                st.json(seg)
                if st.button("Delete Segment", key=f"del_{seg['segmentId']}"): st.session_state.segments = [s for s in st.session_state.segments if s['segmentId'] != seg['segmentId']]; st.rerun()

    if st.session_state.metadata and st.session_state.speakers:
        # ... (rest of the page is unchanged and can be copied from your version) ...
        final_json = {"type": st.session_state.metadata['type'],"value": {"languages": [st.session_state.metadata['internalLanguageCode']],**st.session_state.metadata,"speakers": st.session_state.speakers,"segments": st.session_state.segments,"taskStatus": {"segmentation": {"workflowStatus": "COMPLETE", "workflowType": "LABEL"},"speakerId": {"workflowStatus": "COMPLETE", "workflowType": "LABEL"},"transcription": {"workflowStatus": "COMPLETE", "workflowType": "LABEL"}}}}
        st.subheader("Live JSON Editor"); edited_json_string = st.text_area("JSON Data", json.dumps(final_json, indent=4), height=600, key="json_editor")
        if st.button("Apply JSON Changes"):
            try:
                edited_data = json.loads(edited_json_string); value_section = edited_data.get('value', {}); st.session_state.speakers = value_section.get('speakers', []); st.session_state.segments = value_section.get('segments', []); st.success("JSON changes applied!"); st.rerun()
            except json.JSONDecodeError as e: st.error(f"Invalid JSON format: {e}")
        st.subheader("Download Final Annotation"); st.markdown(get_json_download_link(final_json, "annotated_data.json"), unsafe_allow_html=True)

# =====================================================================================
# MAIN APP ROUTER
# =====================================================================================
if st.session_state.page_state == 'metadata_input':
    metadata_form()
elif st.session_state.page_state == 'annotation':
    if st.sidebar.button("⬅️ Back to Metadata"):
        st.session_state.page_state = 'metadata_input'
        st.rerun()
    annotation_page()
