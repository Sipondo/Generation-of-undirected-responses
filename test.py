import youtube_dl
import re

ydl_opts = {
    'format': 'bestaudio/best',
    'default_search': 'ytsearch',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'opus',
        'preferredquality': '192',
    }],
    'outtmpl': '%(playlist_index)s_%(title)s.%(ext)s',
    'playlistend' : 5,
}

ydl = youtube_dl.YoutubeDL(ydl_opts)

query = 'sir sly'

with ydl:
    result = ydl.extract_info(
        query,
        download=False # We just want to extract the info
    )

if 'entries' in result:
    # Can be a playlist or a list of videos
    video = result['entries'][0]
else:
    # Just a video
    video = result

url_dest = video['webpage_url'].split("?v=")[-1]

url_template = "https://www.youtube.com/watch?v=X&list=RDMMX&start_radio=1"

url = re.sub('X',url_dest,url_template)

with ydl:
    result = ydl.download([url])

# with ydl:
#     result = ydl.extract_info(
#         url,
#         download=False # We just want to extract the info
#     )

ydl_opts = {
    'format': 'bestaudio/best',
    'default_search': 'ytsearch5',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'opus',
        'preferredquality': '192',
    }],
    'outtmpl': '%(playlist_index)s_%(title)s.%(ext)s',
}
ydl = youtube_dl.YoutubeDL(ydl_opts)
query = 'sir sly'
ydl.
with ydl:
    result = ydl.download([query])



class youtubeDownloader():
