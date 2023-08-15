import streamlit as st
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


def music_gen(theme: str, duration: int):
    model = MusicGen.get_pretrained("small")
    model.set_generation_params(duration=duration)  # durationの設定で、musicの長さを設定

    # リストに生成するmusicのテキストを入力する
    # 同じ単語でも同じmusicを生成しない。
    descriptions = [theme]

    wav = model.generate(descriptions)

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f"{idx}", one_wav.cpu(), model.sample_rate, strategy="loudness")


def main():
    st.title("音楽生成アプリ")
    # テキストボックス
    with st.form(key="input_form"):
        # テーマ入力
        theme = st.text_input("テーマを入力してください。", "RPGの戦闘用BGM")
        # 長さ入力
        duration = st.number_input("音楽の長さを入力してください。（単位：ms）", 1, 24, 3)
        # 送信ボタン
        submit_button = st.form_submit_button(label="実行")

    if submit_button:
        music_gen(theme, duration)
        st.audio("0.wav")


if __name__ == "__main__":
    main()
