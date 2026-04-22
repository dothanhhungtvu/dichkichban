import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import zipfile
import io

# Tải cấu hình từ biến môi trường (nếu có .env)
load_dotenv()

st.set_page_config(page_title="Dịch Kịch Bản Cùng AI", page_icon="🎬", layout="wide")

# CSS tùy chỉnh để làm đẹp giao diện
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .title-box {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
        margin-bottom: 30px;
    }
    .title-box h1 {
        color: white !important;
        font-size: 2rem;
        margin-bottom: 5px;
    }
    .title-box p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .file-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-waiting { background: #ffeaa7; color: #2d3436; }
    .status-processing { background: #74b9ff; color: #2d3436; }
    .status-done { background: #55efc4; color: #2d3436; }
    .status-error { background: #ff7675; color: white; }
    
    div[data-testid="stExpander"] {
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title-box">'
    '<h1>🎬 Máy Dịch Kịch Bản Hoạt Hình Thần Kỳ 🚀</h1>'
    '<p>Tải lên tối đa <b>5 file TXT</b> cùng lúc — xử lý song song siêu tốc! Chọn ngôn ngữ dịch: 🇻🇳 Việt / 🇺🇸 English</p>'
    '</div>', 
    unsafe_allow_html=True
)

# KHU VỰC SIDEBAR
with st.sidebar:
    st.header("⚙️ Cấu hình API")
    st.info("Ứng dụng sử dụng **OpenRouter.ai** để gọi mô hình AI siêu thông minh.")
    
    api_key_input = st.text_input(
        "🔑 Nhập Beeknoee API Key của bạn:", 
        type="password", 
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Lấy tại Beeknoee"
    )
    
    model_choice = st.selectbox("🤖 Chọn mô hình AI:", [
        "gemini-2.5-flash-lite",
        "Tự nhập model ID (Custom)"
    ], help="Gemini 2.5 Flash rất lý tưởng cho các câu thoại vì tốc độ phản hồi cực nhanh!")
    
    if model_choice == "Tự nhập model ID (Custom)":
        final_model_choice = st.text_input("✍️ Nhập ID mô hình:", help="Ví dụ: anthropic/claude-3.5-sonnet")
    else:
        final_model_choice = model_choice
    
    st.markdown("---")
    st.markdown("💡 **Mẹo nhỏ:** Tải lên tối đa 5 file TXT cùng lúc, AI sẽ dịch song song tất cả!")


# CÁC HÀM XỬ LÝ (FUNCTIONS)
def clean_script(text):
    """Làm sạch kịch bản bằng Regular Expressions"""
    # Xoá dòng quảng cáo của TurboScribe
    text = text.replace("(Transcribed by TurboScribe. Go Unlimited to remove this message.)", "")
    
    # Xoá các mốc thời gian, vd: [00:00], 00:00:00, [01:23:45]
    text = re.sub(r'\[?\d{2}:\d{2}(:\d{2})?\]?', '', text)
    
    # Xoá các nhãn người nói tiếng Trung/tiếng Anh, VD: [Speaker 1]:, Người nói 1:, 说话人1:
    text = re.sub(r'\[?(?i:speaker|người nói|说话人)\s*\d*\]?:?', '', text)
    
    # Gom khoảng trắng thừa và ngắt dòng quá dãn
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def format_for_tts(text):
    """
    Chuẩn hóa dấu câu để công cụ TTS ngắt nghỉ nhịp nhàng.
    Mỗi dấu chấm (.), chấm hỏi (?), chấm than (!) sẽ được rớt thành dòng mới rõ ràng hơn.
    """
    # Thêm dòng mới sau mỗi dấu chấm câu kết thúc (nếu đang ở giữa dòng)
    text = re.sub(r'([.?!])(\s+)', r'\1\n', text)
    
    # Loại bỏ các dòng trống dư thừa
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Để giúp máy đọc có nhịp thở tốt, nên để cách 2 dòng giữa các câu thoại ngắn
    return "\n\n".join(lines)

def translate_script(kịch_bản, api_key, model, target_language="vi"):
    """Gọi LLM dịch thuật qua Beeknoee. target_language: 'vi' hoặc 'en'"""
    try:
        client = OpenAI(
            base_url="https://platform.beeknoee.com/api/v1/chat/completions",
            api_key=api_key,
        )
        
        if target_language == "en":
            system_prompt = (
                "You are a professional dubbing translator specializing in children's animated films. "
                "Translate the following Chinese dialogue script into English. "
                "The tone must be PLAYFUL, CUTE, and natural — suitable for short TikTok animated videos for kids. "
                "ALWAYS remember: Do NOT add any extra notes or annotations. Return only the pure dialogue content "
                "so that a Text-To-Speech (TTS) engine can read it smoothly. "
                "Do not explain, do not greet."
            )
            user_msg = "Please translate the following script:\n\n" + kịch_bản
        else:
            system_prompt = (
                "Bạn là một biên dịch viên chuyên lồng tiếng phim hoạt hình trẻ em. "
                "Hãy dịch đoạn hội thoại kịch bản tiếng Trung sau sang tiếng Việt. "
                "Văn phong cần VUI NHỘN, ĐÁNG YÊU, ngôn từ tự nhiên, phù hợp với video phim hoạt hình ngắn TikTok dành cho trẻ em. "
                "LUÔN LUÔN ghi nhớ: Tuyệt đối không thêm các chú thích thừa, chỉ trả về nội dung câu thoại thuần túy để máy đọc Text-To-Speech (TTS) có thể đọc mượt mà nhất. "
                "Tuyệt đối không giải thích, không xin chào."
            )
            user_msg = "Hãy dịch kịch bản sau:\n\n" + kịch_bản
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.8
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"🚨 LỖI TỪ API: {str(e)}"

def process_single_file(file_name, raw_text, api_key, model, target_language="vi"):
    """Xử lý 1 file: làm sạch → dịch → format TTS. Trả về dict kết quả."""
    cleaned = clean_script(raw_text)
    raw_translation = translate_script(cleaned, api_key, model, target_language)
    
    if raw_translation.startswith("🚨"):
        return {
            "file_name": file_name,
            "cleaned": cleaned,
            "translated": raw_translation,
            "status": "error",
            "language": target_language
        }
    
    formatted = format_for_tts(raw_translation)
    return {
        "file_name": file_name,
        "cleaned": cleaned,
        "translated": formatted,
        "status": "done",
        "language": target_language
    }


# MAIN LAYOUT: UPLOAD 5 FILES
st.markdown("### 📂 Tải lên kịch bản (tối đa 5 file TXT)")

uploaded_files = st.file_uploader(
    "Kéo thả hoặc chọn tối đa 5 file TXT cùng lúc", 
    type=["txt", "csv"],
    accept_multiple_files=True,
    help="Hỗ trợ file .txt và .csv. Tối đa 5 file mỗi lần."
)

# Giới hạn 5 file
if uploaded_files and len(uploaded_files) > 5:
    st.error("⚠️ Chỉ được tải lên tối đa **5 file** mỗi lần! Bạn đã tải " + str(len(uploaded_files)) + " file.")
    uploaded_files = uploaded_files[:5]

# Khởi tạo session state cho kết quả
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'target_language' not in st.session_state:
    st.session_state.target_language = "vi"

# ===== CHỌN NGÔN NGỮ DỊCH =====
st.markdown("### 🌐 Chọn ngôn ngữ dịch")
st.markdown("""
<style>
.lang-btn-row { display: flex; gap: 16px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

lang_col1, lang_col2, lang_col3 = st.columns([1, 1, 4])
with lang_col1:
    vi_selected = st.session_state.target_language == "vi"
    if st.button(
        "🇻🇳 Tiếng Việt",
        type="primary" if vi_selected else "secondary",
        use_container_width=True,
        key="btn_vi"
    ):
        st.session_state.target_language = "vi"
        st.session_state.processing_done = False
        st.rerun()
with lang_col2:
    en_selected = st.session_state.target_language == "en"
    if st.button(
        "🇺🇸 Tiếng Anh",
        type="primary" if en_selected else "secondary",
        use_container_width=True,
        key="btn_en"
    ):
        st.session_state.target_language = "en"
        st.session_state.processing_done = False
        st.rerun()

# Hiển thị ngôn ngữ đang chọn
if st.session_state.target_language == "vi":
    st.info("✅ Đang dịch sang: **Tiếng Việt** 🇻🇳 — Văn phong vui nhộn, đáng yêu cho TTS.")
else:
    st.info("✅ Translating to: **English** 🇺🇸 — Playful, cute tone for children's TTS.")  

# Hiển thị danh sách file đã upload
if uploaded_files:
    st.markdown(f"**📋 Đã tải lên {len(uploaded_files)} file:**")
    file_cols = st.columns(min(len(uploaded_files), 5))
    for i, f in enumerate(uploaded_files):
        with file_cols[i]:
            size_kb = len(f.getvalue()) / 1024
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); 
                        border: 1px solid rgba(102,126,234,0.3); border-radius: 12px; padding: 15px; text-align: center;">
                <div style="font-size: 2rem;">📄</div>
                <div style="font-weight: 600; font-size: 0.85rem; word-break: break-all;">{f.name}</div>
                <div style="color: gray; font-size: 0.75rem;">{size_kb:.1f} KB</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    
    # ===== NÚT DỊCH =====
    if st.button("🚀 Dịch Tất Cả Song Song", use_container_width=True, type="primary"):
        # Validate
        if not api_key_input:
            st.warning("⚠️ Nhập OpenRouter API Key ở menu Cấu hình bên trái đã nhé!")
        elif len(api_key_input) < 15:
            st.warning("⚠️ Có vẻ API Key của bạn không hợp lệ.")
        elif not final_model_choice:
            st.warning("⚠️ Vui lòng nhập ID mô hình AI!")
        else:
            # Đọc nội dung tất cả files trước
            file_data = []
            for f in uploaded_files:
                file_ext = f.name.split('.')[-1].lower()
                if file_ext == 'csv':
                    try:
                        df = pd.read_csv(f)
                        text_col = None
                        for col in df.columns:
                            col_str = str(col).lower()
                            if 'text' in col_str or 'nội dung' in col_str or 'content' in col_str or 'script' in col_str:
                                text_col = col
                                break
                        if text_col:
                            raw_text = "\n".join(df[text_col].dropna().astype(str).tolist())
                        else:
                            raw_text = "\n".join(df.iloc[:, 0].dropna().astype(str).tolist())
                    except Exception as e:
                        raw_text = f"🚨 Lỗi đọc CSV: {e}"
                else:
                    raw_text = f.getvalue().decode("utf-8")
                
                file_data.append((f.name, raw_text))
            
            # Xử lý song song 5 luồng
            st.session_state.results = []
            
            progress_bar = st.progress(0, text="⏳ Đang khởi động song song...")
            status_container = st.container()
            
            # Tạo placeholder cho từng file
            with status_container:
                status_texts = []
                for i, (fname, _) in enumerate(file_data):
                    status_texts.append(st.empty())
                    status_texts[i].markdown(f"⏳ **{fname}** — Đang chờ...")
            
            results = [None] * len(file_data)
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit tất cả tasks
                future_to_idx = {}
                selected_lang = st.session_state.target_language
                for idx, (fname, raw_text) in enumerate(file_data):
                    future = executor.submit(
                        process_single_file, 
                        fname, raw_text, 
                        api_key_input, final_model_choice,
                        selected_lang
                    )
                    future_to_idx[future] = idx
                
                # Thu thập kết quả khi hoàn thành
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                        
                        if result["status"] == "done":
                            status_texts[idx].markdown(f"✅ **{result['file_name']}** — Hoàn thành!")
                        else:
                            status_texts[idx].markdown(f"❌ **{result['file_name']}** — Lỗi!")
                    except Exception as e:
                        results[idx] = {
                            "file_name": file_data[idx][0],
                            "cleaned": "",
                            "translated": f"🚨 LỖI: {str(e)}",
                            "status": "error"
                        }
                        status_texts[idx].markdown(f"❌ **{file_data[idx][0]}** — Lỗi: {str(e)}")
                    
                    completed_count += 1
                    progress_bar.progress(
                        completed_count / len(file_data), 
                        text=f"✨ Hoàn thành {completed_count}/{len(file_data)} file..."
                    )
            
            progress_bar.progress(1.0, text="🎉 Tất cả đã hoàn thành!")
            st.session_state.results = results
            st.session_state.processing_done = True
            st.rerun()

# ===== HIỂN THỊ KẾT QUẢ =====
if st.session_state.processing_done and st.session_state.results:
    st.markdown("---")
    st.markdown("### 🎯 Kết Quả Dịch Thuật")
    
    # Tổng kết trạng thái
    total = len(st.session_state.results)
    done_count = sum(1 for r in st.session_state.results if r and r["status"] == "done")
    error_count = total - done_count
    
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("📁 Tổng file", total)
    with metric_cols[1]:
        st.metric("✅ Thành công", done_count)
    with metric_cols[2]:
        st.metric("❌ Lỗi", error_count)
    
    st.write("")
    
    # Hiển thị từng file kết quả
    for i, result in enumerate(st.session_state.results):
        if result is None:
            continue
            
        output_name = "Kich_Ban_TTS.txt"
        
        status_icon = "✅" if result["status"] == "done" else "❌"
        
        with st.expander(f"{status_icon} File {i+1}: {result['file_name']}", expanded=(i == 0)):
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("**🧹 Kịch bản gốc đã làm sạch:**")
                st.text_area(
                    "Nội dung gốc", 
                    value=result["cleaned"],
                    height=300,
                    disabled=True,
                    key=f"clean_{i}"
                )
            
            with col_right:
                lang_label = "🇻🇳 Kịch bản tiếng Việt (TTS-ready):" if result.get("language", "vi") == "vi" else "🇺🇸 Script in English (TTS-ready):"
                st.markdown(f"**{lang_label}**")
                st.text_area(
                    "Bản dịch", 
                    value=result["translated"],
                    height=300,
                    disabled=True,
                    key=f"trans_{i}"
                )
            
            # Nút tải xuống cho từng file
            if result["status"] == "done":
                dl_label = f"💾 Tải Xuống: {output_name}" if result.get("language", "vi") == "vi" else f"💾 Download: {output_name}"
                st.download_button(
                    label=dl_label,
                    data=result["translated"].encode("utf-8"),
                    file_name=output_name,
                    mime="text/plain",
                    use_container_width=True,
                    key=f"download_{i}"
                )
    
    # Nút tải tất cả (tải về tất cả dưới dạng file ZIP để giữ tên Kich_Ban_TTS.txt)
    if done_count > 1:
        st.markdown("---")
        st.markdown("### 📦 Tải Xuống Tất Cả")
        
        # Tạo file ZIP chứa tất cả các bản dịch, mỗi file nằm trong 1 folder tên gốc
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, result in enumerate(st.session_state.results):
                if result and result["status"] == "done":
                    base_name = os.path.splitext(result['file_name'])[0]
                    zip_path = f"{base_name}/Kich_Ban_TTS.txt"
                    zip_file.writestr(zip_path, result["translated"].encode("utf-8"))
        
        st.download_button(
            label="📥 Tải Về Tất Cả Các File (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="TatCa_KichBan_TTS.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_all_zip"
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 12px;'>"
    "Tạo bởi Full-Stack AI Developer 🎬 Phát triển bằng Streamlit Python. | Xử lý song song 5 luồng ⚡"
    "</p>", 
    unsafe_allow_html=True
)
