#!/bin/bash
# TTS Engine Testing CLI
# Tests EmotiVoice, OpenVoice, and Edge-TTS engines
# Usage: ./test_tts.sh [command] [options]

set -e

PORT=8001
BASE_URL="http://localhost:$PORT/api/tts"
OUTPUT_DIR="/tmp/tts_tests"
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if server is running
check_server() {
    if ! curl -s "$BASE_URL/../health" > /dev/null 2>&1; then
        print_error "memOS server is not running on port $PORT"
        echo "Start with: ./start_server.sh"
        exit 1
    fi
}

# List available TTS engines
list_engines() {
    print_header "Available TTS Engines"
    curl -s "$BASE_URL/engines" | python -m json.tool
}

# List EmotiVoice emotions
list_emotions() {
    print_header "EmotiVoice Available Emotions"
    curl -s "$BASE_URL/emotivoice/emotions" | python -m json.tool
}

# List EmotiVoice speakers
list_speakers() {
    print_header "EmotiVoice Speaker Presets"
    curl -s "$BASE_URL/emotivoice/speakers" | python -m json.tool
}

# List OpenVoice styles
list_styles() {
    print_header "OpenVoice Available Styles"
    curl -s "$BASE_URL/openvoice/styles" | python -m json.tool
}

# Test EmotiVoice synthesis
test_emotivoice() {
    local text="${1:-Hello, I am here to help you through this difficult time.}"
    local emotion="${2:-Empathetic}"
    local speaker="${3:-8051}"
    local output="$OUTPUT_DIR/emotivoice_${emotion,,}.wav"

    print_header "Testing EmotiVoice"
    print_info "Text: $text"
    print_info "Emotion: $emotion"
    print_info "Speaker: $speaker"
    echo ""

    local start_time=$(date +%s.%N)

    local response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/emotivoice/synthesize" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"$text\",\"emotion\":\"$emotion\",\"speaker\":\"$speaker\",\"speed\":1.0,\"format\":\"wav\"}" \
        -o "$output" 2>&1)

    local http_code=$(echo "$response" | tail -1)
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    if [ -f "$output" ] && file "$output" | grep -q "WAVE audio"; then
        local size=$(ls -lh "$output" | awk '{print $5}')
        print_success "Generated: $output ($size) in ${duration}s"
        echo ""
        echo "Play with: aplay $output"
    else
        print_error "Failed to generate audio"
        if [ -f "$output" ]; then
            cat "$output"
        fi
    fi
}

# Test OpenVoice synthesis
test_openvoice() {
    local text="${1:-Hello, I am your friendly companion.}"
    local style="${2:-friendly}"
    local output="$OUTPUT_DIR/openvoice_${style}.wav"

    print_header "Testing OpenVoice"
    print_info "Text: $text"
    print_info "Style: $style"
    echo ""

    local start_time=$(date +%s.%N)

    curl -s -X POST "$BASE_URL/openvoice/synthesize" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"$text\",\"style\":\"$style\",\"speed\":1.0,\"format\":\"wav\"}" \
        -o "$output" 2>&1

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    if [ -f "$output" ] && file "$output" | grep -q "WAVE audio"; then
        local size=$(ls -lh "$output" | awk '{print $5}')
        print_success "Generated: $output ($size) in ${duration}s"
        echo ""
        echo "Play with: aplay $output"
    else
        print_error "Failed to generate audio"
        if [ -f "$output" ]; then
            cat "$output"
        fi
    fi
}

# Test Edge-TTS synthesis
test_edgetts() {
    local text="${1:-Hello, I am the Edge TTS voice.}"
    local voice="${2:-en-US-AriaNeural}"
    local output="$OUTPUT_DIR/edgetts_$(echo $voice | tr ':' '_').wav"

    print_header "Testing Edge-TTS"
    print_info "Text: $text"
    print_info "Voice: $voice"
    echo ""

    local start_time=$(date +%s.%N)

    curl -s "$BASE_URL/base_tts/?text=$(echo "$text" | jq -sRr @uri)&voice=$voice" \
        -o "$output" 2>&1

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    if [ -f "$output" ] && file "$output" | grep -q "WAVE audio"; then
        local size=$(ls -lh "$output" | awk '{print $5}')
        print_success "Generated: $output ($size) in ${duration}s"
        echo ""
        echo "Play with: aplay $output"
    else
        print_error "Failed to generate audio"
        if [ -f "$output" ]; then
            cat "$output"
        fi
    fi
}

# Test all emotions with EmotiVoice
test_all_emotions() {
    print_header "Testing All EmotiVoice Emotions"
    local text="${1:-I understand how you feel.}"

    local emotions=("Happy" "Sad" "Angry" "Empathetic" "Encouraging" "Calm" "Excited" "Gentle" "Soothing")

    for emotion in "${emotions[@]}"; do
        echo -e "${YELLOW}Testing: $emotion${NC}"
        local output="$OUTPUT_DIR/emotivoice_${emotion,,}.wav"

        curl -s -X POST "$BASE_URL/emotivoice/synthesize" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"$text\",\"emotion\":\"$emotion\",\"speed\":1.0,\"format\":\"wav\"}" \
            -o "$output" 2>&1

        if [ -f "$output" ] && file "$output" | grep -q "WAVE audio"; then
            local size=$(ls -lh "$output" | awk '{print $5}')
            print_success "$emotion: $output ($size)"
        else
            print_error "$emotion: Failed"
        fi
    done

    echo ""
    print_info "Audio files saved to: $OUTPUT_DIR/"
}

# Test all styles with OpenVoice
test_all_styles() {
    print_header "Testing All OpenVoice Styles"
    local text="${1:-This is a test of different voice styles.}"

    local styles=("default" "friendly" "cheerful" "excited" "sad" "angry" "whispering")

    for style in "${styles[@]}"; do
        echo -e "${YELLOW}Testing: $style${NC}"
        local output="$OUTPUT_DIR/openvoice_${style}.wav"

        curl -s -X POST "$BASE_URL/openvoice/synthesize" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"$text\",\"style\":\"$style\",\"speed\":1.0,\"format\":\"wav\"}" \
            -o "$output" 2>&1

        if [ -f "$output" ] && file "$output" | grep -q "WAVE audio"; then
            local size=$(ls -lh "$output" | awk '{print $5}')
            print_success "$style: $output ($size)"
        else
            print_error "$style: Failed"
        fi
    done

    echo ""
    print_info "Audio files saved to: $OUTPUT_DIR/"
}

# Unload models to free VRAM
unload_models() {
    print_header "Unloading TTS Models (Free VRAM)"

    local response=$(curl -s -X POST "$BASE_URL/models/unload" \
        -H "Content-Type: application/json" \
        -d '{"engines":["emotivoice","openvoice"]}')

    echo "$response" | python -m json.tool
}

# Check model status
model_status() {
    print_header "TTS Model Status"
    curl -s "$BASE_URL/models/status" | python -m json.tool
}

# Compare all engines
compare_engines() {
    print_header "Comparing All TTS Engines"
    local text="${1:-Hello, this is a comparison test of all text to speech engines.}"

    echo -e "${YELLOW}Text: $text${NC}"
    echo ""

    echo -e "${CYAN}1. Edge-TTS (Cloud)${NC}"
    test_edgetts "$text" "en-US-AriaNeural"
    echo ""

    echo -e "${CYAN}2. EmotiVoice (Local GPU - Emotion Control)${NC}"
    test_emotivoice "$text" "Calm"
    echo ""

    echo -e "${CYAN}3. OpenVoice (Local GPU - Style Control)${NC}"
    test_openvoice "$text" "friendly"
    echo ""

    print_info "All files saved to: $OUTPUT_DIR/"
    echo "Compare with:"
    echo "  aplay $OUTPUT_DIR/edgetts_en-US-AriaNeural.wav"
    echo "  aplay $OUTPUT_DIR/emotivoice_calm.wav"
    echo "  aplay $OUTPUT_DIR/openvoice_friendly.wav"
}

# Register a voice for cloning (OpenVoice)
register_voice() {
    local audio_file="$1"
    local voice_name="${2:-my_voice}"
    local user_id="${3:-test_user}"

    if [ -z "$audio_file" ]; then
        print_error "Usage: $0 register-voice <audio_file> [voice_name] [user_id]"
        echo "Audio file should be 10-30 seconds of clear speech"
        exit 1
    fi

    if [ ! -f "$audio_file" ]; then
        print_error "File not found: $audio_file"
        exit 1
    fi

    print_header "Registering Voice for Cloning"
    print_info "Audio: $audio_file"
    print_info "Name: $voice_name"
    print_info "User: $user_id"
    echo ""

    curl -s -X POST "$BASE_URL/openvoice/register-voice" \
        -F "user_id=$user_id" \
        -F "voice_name=$voice_name" \
        -F "audio_file=@$audio_file" | python -m json.tool
}

# List registered voices
list_voices() {
    local user_id="${1:-}"
    print_header "Registered Voices"

    local url="$BASE_URL/openvoice/voices"
    if [ -n "$user_id" ]; then
        url="$url?user_id=$user_id"
    fi

    curl -s "$url" | python -m json.tool
}

# Show help
show_help() {
    echo "TTS Engine Testing CLI"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  engines              List available TTS engines"
    echo "  emotions             List EmotiVoice emotions"
    echo "  speakers             List EmotiVoice speaker presets"
    echo "  styles               List OpenVoice styles"
    echo ""
    echo "  emotivoice [text] [emotion] [speaker]"
    echo "                       Test EmotiVoice synthesis"
    echo "                       Default emotion: Empathetic"
    echo ""
    echo "  openvoice [text] [style]"
    echo "                       Test OpenVoice synthesis"
    echo "                       Default style: friendly"
    echo ""
    echo "  edgetts [text] [voice]"
    echo "                       Test Edge-TTS synthesis"
    echo "                       Default voice: en-US-AriaNeural"
    echo ""
    echo "  all-emotions [text]  Test all EmotiVoice emotions"
    echo "  all-styles [text]    Test all OpenVoice styles"
    echo "  compare [text]       Compare all engines side-by-side"
    echo ""
    echo "  register-voice <audio_file> [name] [user_id]"
    echo "                       Register voice sample for cloning"
    echo "  voices [user_id]     List registered voices"
    echo ""
    echo "  unload               Unload models to free VRAM"
    echo "  status               Check which models are loaded"
    echo ""
    echo "Examples:"
    echo "  $0 engines"
    echo "  $0 emotivoice 'I understand how you feel' Empathetic"
    echo "  $0 openvoice 'Hello there!' cheerful"
    echo "  $0 all-emotions 'Take your time, I am here for you'"
    echo "  $0 compare 'Testing all engines'"
    echo "  $0 register-voice my_recording.wav 'my_voice' user123"
    echo ""
    echo "Output files are saved to: $OUTPUT_DIR/"
}

# Main command handler
case "${1:-help}" in
    engines)
        check_server
        list_engines
        ;;
    emotions)
        check_server
        list_emotions
        ;;
    speakers)
        check_server
        list_speakers
        ;;
    styles)
        check_server
        list_styles
        ;;
    emotivoice)
        check_server
        test_emotivoice "$2" "$3" "$4"
        ;;
    openvoice)
        check_server
        test_openvoice "$2" "$3"
        ;;
    edgetts|edge-tts)
        check_server
        test_edgetts "$2" "$3"
        ;;
    all-emotions)
        check_server
        test_all_emotions "$2"
        ;;
    all-styles)
        check_server
        test_all_styles "$2"
        ;;
    compare)
        check_server
        compare_engines "$2"
        ;;
    register-voice)
        check_server
        register_voice "$2" "$3" "$4"
        ;;
    voices)
        check_server
        list_voices "$2"
        ;;
    unload)
        check_server
        unload_models
        ;;
    status)
        check_server
        model_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
