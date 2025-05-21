import yt_dlp

VIDEO_URL = "https://www.youtube.com/watch?v=ignysw4pFO0&t=25s"

# We'll try known single-file formats first (e.g. 22 = 720p mp4, 18 = 360p mp4),
# so yt_dlp doesn't need to merge anything.
ydl_opts = {
    'format': '22/18/best[ext=mp4]',  # 22 is 720p mp4+audio, fallback 18 is 360p mp4+audio, else best mp4
    # Customize output filename as you like:
    'outtmpl': r'C:\Users\97254\PycharmProjects\pythonProject\downloads\%(title)s.%(ext)s'
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([VIDEO_URL])
