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

def transcribe_audio_segment_with_gemini(full_audio_bytes, start_time, end_time, api_key):
    """
    Extracts the audio segment between start_time and end_time (in seconds) and sends only that segment to Gemini for transcription.
    """
    try:
        # pydub can often auto-detect the format from the bytes
        audio = AudioSegment.from_file(io.BytesIO(full_audio_bytes))
        # Gemini expects wav, so we will export as wav regardless of input format
        mime_type = "audio/wav"

        # Convert start/end to milliseconds
        start_ms = int(float(start_time) * 1000)
        end_ms = int(float(end_time) * 1000)
        segment = audio[start_ms:end_ms]

        # Export segment to WAV in memory
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        segment_bytes = buf.getvalue()
        audio_base64 = base64.b64encode(segment_bytes).decode()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        headers = {
            'Content-Type': 'application/json'
        }

        duration = end_time - start_time
        prompt = f"""You are an expert audio transcriptionist. Your task is to process an audio file and transcribe ONLY a specific time segment.\n\nIMPORTANT INSTRUCTIONS:\n1.  Analyze ONLY the audio content from 0.000 seconds to {duration:.3f} seconds.\n2.  The duration of the target segment is {duration:.3f} seconds.\n3.  You MUST IGNORE all audio content before 0.000 seconds and after {duration:.3f} seconds.\n4.  Focus exclusively on the specified time range: 0.000s - {duration:.3f}s.\n5.  Automatically detect the language spoken *within that specific segment*.\n6.  Provide a highly accurate transcription of ONLY that time segment.\n7.  If there is no audible speech in that specific time range, you MUST respond with the exact text \"[SILENCE]\".\n8.  If there is only background noise, music, or non-speech sounds in that segment, you MUST respond with the exact text \"[NOISE]\".\n9.  Return ONLY the final transcribed text from the specified segment. Do not include any commentary, timestamps, or introductory phrases like \"Here is the transcription:\".\n"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2000,
                "topP": 0.8,
                "topK": 40
            }
        }

        with st.spinner(f"Requesting transcription for segment {start_time}s - {end_time}s..."):
            response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            if not result.get('candidates'):
                st.error(f"Gemini API Error: No candidates returned. Response: {result}")
                return None
            content = result['candidates'][0].get('content', {})
            parts = content.get('parts', [])
            if parts:
                transcription = parts[0].get('text', '').strip()
                return transcription
            return "[NO_CONTENT]"
        else:
            error_msg = f"Gemini API Error: {response.status_code}"
            if response.text:
                error_msg += f" - {response.text}"
            st.error(error_msg)
            return None

    except requests.exceptions.Timeout:
        st.error("Request timed out. The audio file might be too large or there could be network issues.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during transcription: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# =====================================================================================
# CUSTOM AUDIO PLAYER COMPONENT (Unchanged)
# =====================================================================================

def audio_player_component(audio_bytes: bytes):
    """
    Creates a custom audio player component using wavesurfer.js that displays
    the current time, allows click-to-seek, and has playback speed control.
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
                    <option value="0.5">0.5x</option>
                    <option value="1" selected>1.0x</option>
                    <option value="1.5">1.5x</option>
                    <option value="2">2.0x</option>
                </select>
            </div>
        </div>
    </div>
    <script src="https://unpkg.com/wavesurfer.js"></script>
    <script>
        var wavesurfer = WaveSurfer.create({{
            container: '#waveform',
            waveColor: 'violet',
            progressColor: 'purple',
            barWidth: 2,
            barRadius: 3,
            height: 100,
            barGap: 3,
            responsive: true,
            fillParent: true,
            minPxPerSec: 1,
            cursorWidth: 1,
            cursorColor: 'purple'
        }});
        wavesurfer.load('data:audio/wav;base64,{b64_audio}');

        const playBtn = document.getElementById('playBtn');
        const timeDisplay = document.getElementById('time-display');
        const speedSelector = document.getElementById('playbackSpeed');

        playBtn.onclick = function() {{ wavesurfer.playPause(); }};

        wavesurfer.on('audioprocess', function() {{
            let currentTime = wavesurfer.getCurrentTime();
            timeDisplay.textContent = currentTime.toFixed(3);
        }});
        
        wavesurfer.on('interaction', function() {{
            let currentTime = wavesurfer.getCurrentTime();
            timeDisplay.textContent = currentTime.toFixed(3);
        }});

        speedSelector.onchange = function() {{
            wavesurfer.setPlaybackRate(this.value);
        }};
        
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
            
            speakers_input.append({
                "speakerId": speaker_id if speaker_id else str(uuid.uuid4()),
                "gender": gender, "genderSource": gender_source, "speakerNativity": speaker_nativity,
                "speakerNativitySource": speaker_nativity_source, "speakerRole": speaker_role,
                "speakerRoleSource": speaker_role_source,
                "languages": [language_locale]
            })

            if i == 0:
                 speaker_dominant_varieties_data.append({
                     "languageLocale": language_locale, 
                     "languageVariety": [v.strip() for v in language_variety.split(",") if v.strip()], 
                     "otherLanguageInfluence": [v.strip() for v in other_language_influence.split(",") if v.strip()]
                 })
            
        submit_button = st.form_submit_button(label="Save Metadata and Proceed to Annotation")
        if submit_button:
            st.session_state.metadata = {
                "type": {"name": type_name, "version": type_version},
                "languageInfo": {
                    "spokenLanguages": [lang_full],
                    "speakerDominantVarieties": speaker_dominant_varieties_data
                },
                "domainInfo": {"domainVersion": "1.0", "domainList": [{"domain": domain_name, "topicList": [t.strip() for t in topic_list.split(',')]}]},
                "annotatorInfo": {"loginEncrypted": login_encrypted, "annotatorId": annotator_id},
                "conventionInfo": {"masterConventionName": master_convention, "customAddendum": custom_addendum},
                "internalLanguageCode": lang_short
            }
            st.session_state.speakers = speakers_input
            st.session_state.page_state = 'annotation'
            st.success("Metadata saved successfully! Proceed to the annotation step below.")
            st.rerun()

# =====================================================================================
# PAGE 2: AUDIO ANNOTATION AND JSON EDITOR (UPDATED WITH AUDIO METRICS)
# =====================================================================================

def annotation_page():
    st.title("Step 2: Audio Annotation")

    st.markdown("""
        <style>
        div[data-testid="stElementContainer"]:has(iframe[title^="st.iframe"]) {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])

    if uploaded_file:
        if st.session_state.current_audio is None or st.session_state.current_audio.get('name') != uploaded_file.name:
            st.session_state.current_audio = {
                'name': uploaded_file.name,
                'bytes': uploaded_file.getvalue()
            }

        audio_bytes = st.session_state.current_audio['bytes']

        # ================================================================== #
        # ========= NEW: CALCULATE AND DISPLAY AUDIO METRICS HERE ========== #
        # ================================================================== #
        st.subheader("Audio File Properties")
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # --- Calculate Metrics ---
            duration_seconds = len(audio_segment) / 1000.0
            peak_loudness_dbfs = audio_segment.max_dBFS
            sample_rate_khz = audio_segment.frame_rate / 1000.0
            channels = "Stereo" if audio_segment.channels >= 2 else "Mono"
            
            # --- Display Metrics in Columns ---
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Duration", value=f"{duration_seconds:.2f} s")
            with col2:
                st.metric(label="Peak Loudness", value=f"{peak_loudness_dbfs:.2f} dBFS")
            with col3:
                st.metric(label="Sample Rate", value=f"{sample_rate_khz:.1f} kHz")
            with col4:
                st.metric(label="Channels", value=channels)

        except Exception as e:
            st.error(f"Could not process audio file to get properties. Error: {e}")
        # ================================================================== #
        # ======================= END OF NEW SECTION ======================= #
        # ================================================================== #

        st.subheader("Audio Player")
        audio_player_component(audio_bytes)

        st.subheader("Add a New Segment")
        st.markdown("Use the player above to find start/end times, then enter the details below.")

        time_col1, time_col2, transcribe_col = st.columns([2, 2, 1])

        with time_col1:
            start_time = st.text_input("Start Time (seconds.milliseconds, e.g., 0.0)", "0.0", key="start_time_input")
        with time_col2:
            end_time = st.text_input("End Time (seconds.milliseconds, e.g., 14.711)", "5.000", key="end_time_input")

        with transcribe_col:
            st.markdown("<br>", unsafe_allow_html=True)
            transcribe_button = st.button("🎙️ Transcribe", help="Transcribe this audio segment using Gemini")

        if transcribe_button:
            try:
                start_float = float(start_time)
                end_float = float(end_time)

                if start_float >= end_float:
                    st.error("Start time must be less than end time!")
                elif start_float < 0:
                    st.error("Start time cannot be negative!")
                else:
                    try:
                        api_key = st.secrets["GEMINI_API_KEY"]
                    except (FileNotFoundError, KeyError):
                        st.error("GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
                        st.stop()
                    
                    transcription = transcribe_audio_segment_with_gemini(
                        audio_bytes, start_float, end_float, api_key
                    )

                    if transcription is not None:
                        if transcription == "[SILENCE]":
                            st.info("No speech detected in the specified time segment.")
                            st.session_state.transcription_content = ""
                        elif transcription == "[NOISE]":
                            st.info("Only background noise/music detected in the segment.")
                            st.session_state.transcription_content = "[NOISE]"
                        elif transcription == "[NO_CONTENT]":
                             st.warning("No content was returned from the API for this segment.")
                             st.session_state.transcription_content = ""
                        else:
                            st.session_state.transcription_content = transcription
                            st.success(f"✅ Transcribed segment ({start_float}s - {end_float}s) successfully!")
                        st.rerun()
                    else:
                        pass 
            except ValueError:
                st.error("Please enter valid numeric values for start and end times")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

        with st.form(key="segment_form", clear_on_submit=True):
            primary_type = st.selectbox("Primary Type", ["Speech", "Noise", "Music", "Silence"], index=0)
            loudness_level = st.selectbox("Loudness Level", ["Normal", "Quiet", "Loud"], index=0)

            transcription = st.text_area("Transcription Content", 
                                       value=st.session_state.transcription_content, 
                                       help="Use the 'Transcribe' button above to auto-fill this field")
            
            if st.session_state.speakers:
                speaker_options = {s['speakerId']: f"Speaker {i+1} ({s.get('speakerRole', 'N/A')})" for i, s in enumerate(st.session_state.speakers)}
                selected_speaker_id = st.selectbox("Speaker", options=list(speaker_options.keys()), format_func=lambda x: speaker_options[x])
            else:
                st.warning("No speakers defined. Please add speakers in the metadata step.")
                selected_speaker_id = None

            add_segment_button = st.form_submit_button("Add Segment")

            if add_segment_button:
                if selected_speaker_id is None:
                    st.error("Cannot add segment. Please define at least one speaker in the metadata.")
                else:
                    try:
                        start_float = float(start_time)
                        end_float = float(end_time)
                        
                        if start_float >= end_float:
                            st.error("Start time must be less than end time!")
                        else:
                            lang_code = st.session_state.metadata.get('internalLanguageCode', 'en_US')
                            new_segment = {
                                "start": start_float,
                                "end": end_float,
                                "segmentId": str(uuid.uuid4()),
                                "primaryType": primary_type,
                                "loudnessLevel": loudness_level,
                                "language": lang_code,
                                "segmentLanguages": [lang_code],
                                "speakerId": selected_speaker_id,
                                "transcriptionData": {"content": transcription}
                            }
                            st.session_state.segments.append(new_segment)
                            st.session_state.transcription_content = ""
                            st.success(f"Segment ({primary_type}) added successfully!")
                            st.rerun()
                    except ValueError:
                        st.error("Please enter valid numeric values for start and end times")

    if st.session_state.segments:
        st.subheader("Annotated Segments")
        st.markdown("You can delete segments here. For detailed edits, use the JSON editor below.")
        
        sorted_segments = sorted(st.session_state.segments, key=lambda x: float(x.get('start', 0)))
        st.session_state.segments = sorted_segments

        for i, seg in enumerate(st.session_state.segments):
            with st.expander(f"Segment {i+1}: {seg['start']} - {seg['end']} ({seg['primaryType']})"):
                st.json(seg)
                if st.button("Delete Segment", key=f"del_{seg['segmentId']}"):
                    st.session_state.segments = [
                        s for s in st.session_state.segments if s['segmentId'] != seg['segmentId']
                    ]
                    st.rerun()

    if st.session_state.metadata and st.session_state.speakers:
        final_json = {
            "type": st.session_state.metadata['type'],
            "value": {
                "languages": [st.session_state.metadata['internalLanguageCode']],
                "languageInfo": st.session_state.metadata['languageInfo'],
                "domainInfo": st.session_state.metadata['domainInfo'],
                "conventionInfo": st.session_state.metadata['conventionInfo'],
                "annotatorInfo": st.session_state.metadata['annotatorInfo'],
                "speakers": st.session_state.speakers,
                "segments": st.session_state.segments,
                "taskStatus": {"segmentation": {"workflowStatus": "COMPLETE", "workflowType": "LABEL"}, "speakerId": {"workflowStatus": "COMPLETE", "workflowType": "LABEL"}, "transcription": {"workflowStatus": "COMPLETE", "workflowType": "LABEL"}}
            }
        }
        st.subheader("Live JSON Editor")
        st.markdown("You can directly edit, update, or delete values in the JSON below. Click 'Apply Changes' to save them to the application state.")
        json_string = json.dumps(final_json, indent=4)
        edited_json_string = st.text_area(label="JSON Data", value=json_string, height=600, key="json_editor")
        
        apply_button = st.button("Apply JSON Changes")
        if apply_button:
            try:
                edited_data = json.loads(edited_json_string)
                value_section = edited_data.get('value', {})
                st.session_state.speakers = value_section.get('speakers', [])
                st.session_state.segments = value_section.get('segments', [])
                st.success("JSON changes have been applied!")
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {e}")

        st.subheader("Download Final Annotation")
        st.markdown(get_json_download_link(final_json, filename="annotated_data.json"), unsafe_allow_html=True)

# =====================================================================================
# MAIN APP ROUTER (Unchanged)
# =====================================================================================

if st.session_state.page_state == 'metadata_input':
    metadata_form()
elif st.session_state.page_state == 'annotation':
    if st.sidebar.button("⬅️ Back to Metadata"):
        st.session_state.page_state = 'metadata_input'
        st.rerun()
    annotation_page()
