import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import os
from dotenv import load_dotenv

# Tải cấu hình từ biến môi trường (nếu có .env)
load_dotenv()

st.set_page_config(page_title="Dịch Kịch Bản Cùng AI", page_icon="🎬", layout="wide")

# CSS tùy chỉnh để làm đẹp giao diện
st.markdown("""
<style>
    .title-box {
        text-align: center;
        background: linear-gradient(135deg, #FF9A9E 0%, #FECFEF 99%, #FECFEF 100%);
        padding: 25px;
        border-radius: 15px;
        color: #d11a2a;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff2e2e !important;
        transform: scale(1.02);
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title-box">'
    '<h1>🎬 Máy Dịch Kịch Bản Hoạt Hình Thần Kỳ 🚀</h1>'
    '<p>Biến kịch bản tiếng Trung cứng nhắc thành câu chuyện cực vui nhộn cho bé!</p>'
    '</div>', 
    unsafe_allow_html=True
)

# KHU VỰC SIDEBAR
with st.sidebar:
    st.header("⚙️ Cấu hình API")
    st.info("Ứng dụng sử dụng **OpenRouter.ai** để gọi mô hình AI siêu thông minh.")
    
    api_key_input = st.text_input(
        "🔑 Nhập OpenRouter API Key của bạn:", 
        type="password", 
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Lấy tại https://openrouter.ai/keys"
    )
    
    model_choice = st.selectbox("🤖 Chọn mô hình AI:", [
        "openai/gpt-4o-mini",
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-lite-001",
        "openai/gpt-5-nano",
        "openai/gpt-4.1-nano",
        "meta-llama/llama-3.2-3b-instruct:free",
    ], help="Gemini 2.5 Flash rất lý tưởng cho các câu thoại vì tốc độ phản hồi cực nhanh!")
    
    st.markdown("---")
    st.markdown("💡 **Mẹo nhỏ:** Kịch bản sau khi làm sạch sẽ tự động mất đi các tên nhân vật và mốc thời gian thừa!")

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

def translate_script(kịch_bản, api_key, model):
    """Gọi LLM dịch thuật qua OpenRouter"""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        system_prompt = (
            "Bạn là một biên dịch viên chuyên lồng tiếng phim hoạt hình trẻ em. "
            "Hãy dịch đoạn hội thoại kịch bản tiếng Trung sau sang tiếng Việt. "
            "Văn phong cần VUI NHỘN, ĐÁNG YÊU, ngôn từ tự nhiên, phù hợp với video phim hoạt hình ngắn TikTok dành cho trẻ em. "
            "LUÔN LUÔN ghi nhớ: Tuyệt đối không thêm các chú thích thừa, chỉ trả về nội dung câu thoại thuần túy để máy đọc Text-To-Speech (TTS) có thể đọc mượt mà nhất. "
            "Tuyệt đối không giải thích, không xin chào."
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hãy dịch kịch bản sau:\n\n" + kịch_bản}
            ],
            temperature=0.8 # Tạo độ sáng tạo, mượt mà và tự nhiên cho văn nói
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"🚨 LỖI TỪ API: {str(e)}"

# MAIN LAYOUT: STATE MANAGEMENT VÀ UPLOAD
if 'cleaned_text' not in st.session_state:
    st.session_state.cleaned_text = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""

uploaded_file = st.file_uploader(
    "📥 Cập nhật kịch bản của bạn (Định dạng .txt hoặc .csv phân tách dòng)", 
    type=["txt", "csv"]
)

# TIẾN HÀNH ĐỌC FILE KHI CÓ FILE UPLOAD
if uploaded_file is not None:
    with st.spinner("Đang chép nội dung file..."):
        file_ext = uploaded_file.name.split('.')[-1].lower()
        raw_text = ""
        
        if file_ext == 'csv':
            try:
                # Cố gắng đọc định dạng CSV
                df = pd.read_csv(uploaded_file)
                # Tìm cột có khả năng chứa text
                text_col = None
                for col in df.columns:
                    col_str = str(col).lower()
                    if 'text' in col_str or 'nội dung' in col_str or 'content' in col_str or 'script' in col_str:
                        text_col = col
                        break
                
                # Nếu tìm thấy cột trùng khớp lý tưởng
                if text_col:
                    raw_text = "\n".join(df[text_col].dropna().astype(str).tolist())
                # Không thì lấy dữ liệu cột đầu tiên
                else:
                    raw_text = "\n".join(df.iloc[:, 0].dropna().astype(str).tolist())
            except Exception as e:
                st.error(f"Lỗi đọc file CSV: {e}")
        else: # TXT
            raw_text = uploaded_file.getvalue().decode("utf-8")
        
        # Làm sạch kịch bản ngay lập tức và đưa vào state
        st.session_state.cleaned_text = clean_script(raw_text)

# BỐ CỤC 2 CỘT
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧹 Kịch Bản Gốc Đã Làm Sạch", divider="red")
    st.text_area(
        "Nội dung này đã được bỏ timestamps và tên Speaker:", 
        value=st.session_state.cleaned_text, 
        height=450, 
        disabled=True, 
        key="clean_area"
    )
    
    # Nút bấm trung tâm
    if st.session_state.cleaned_text:
        # Tách dòng trống để nút hiển thị đẹp
        st.write("") 
        if st.button("🚀 Bắt Đầu Nhập Vai Dịch Thuật", use_container_width=True):
            if not api_key_input:
                st.warning("⚠️ Nhập OpenRouter API Key ở menu Cấu hình bên trái đã nhé!")
            elif len(api_key_input) < 15:
                st.warning("⚠️ Có vẻ API Key của bạn không hợp lệ.")
            else:
                with st.spinner("✨ AI phản hồi cực gắt... Xin vui lòng đợi! 🪄"):
                    # Dịch thuật
                    raw_translation = translate_script(
                        st.session_state.cleaned_text, 
                        api_key_input,
                        model_choice
                    )
                    # Format đẹp nhịp nghỉ cho máy ảo TTS
                    st.session_state.translated_text = format_for_tts(raw_translation)

with col2:
    st.subheader("🇻🇳 Kịch Bản Tiếng Việt (Output)", divider="rainbow")
    st.text_area(
        "Nội dung này đã sẵn sàng để gắn vào Video, TTS:", 
        value=st.session_state.translated_text, 
        height=450,
        key="translated_area"
    )
    
    # Hiển thị nút download nếu dịch chuẩn (không dính lỗi)
    if st.session_state.translated_text and not st.session_state.translated_text.startswith("🚨"):
        st.write("")
        st.download_button(
            label="💾 Tải Xuống File TXT Chuẩn Text-to-Speech",
            data=st.session_state.translated_text.encode("utf-8"),
            file_name="Kich_Ban_TTS.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 12px;'>"
    "Tạo bởi Full-Stack AI Developer 🎬 Phát triển bằng Streamlit Python."
    "</p>", 
    unsafe_allow_html=True
)
