"""
Daily video pipeline for a kids' comedy channel.

1. Claude invents an original topic and writes a short scene-by-scene script.
2. Claude checks the script as a safety filter.
3. OpenAI draws each scene and narrates it.
4. FFmpeg joins the scenes into one 720p MP4.
5. The video is uploaded to YouTube as public and marked made-for-kids.
"""

import base64
import json
import os
import subprocess
from pathlib import Path

import anthropic
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CLAUDE_MODEL = "claude-sonnet-5-5"
HISTORY_FILE = Path("topics_history.json")
PAUSE_FILE = Path("PAUSED")
WORK = Path("work")
MAX_SCENES = 12


def openai_headers():
    return {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}


def ask_claude_json(prompt, max_tokens=2000):
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    # Parse the outermost JSON object, ignoring any stray text around it.
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def make_script(past_titles):
    prompt = f"""You write short comedy videos for children and their families.

Invent one fresh, original topic for today's video. It must not resemble any of these past titles:
{json.dumps(past_titles[-60:])}

Rules:
- Make it genuinely funny for kids and grown-ups alike: silly situations, playful characters, a punchline in most scenes.
- Original characters only. No trademarked or copyrighted characters, brands, or songs.
- Clean, kind humor. Nothing scary, violent, sexual, hateful, political, or about real people.
- Use 8 to 10 scenes. Each scene has one or two sentences of narration that sound natural read aloud.
- Each scene also has an image_prompt describing a bright cartoon illustration with no text in it.

Return only a JSON object with these keys:
"title" (under 70 characters),
"description" (two or three short paragraphs),
"tags" (8 to 12 short strings),
"scenes" (a list of objects, each with "narration" and "image_prompt").
"""
    return ask_claude_json(prompt, max_tokens=3000)


def passes_safety_review(script):
    prompt = f"""Review this script for a children's comedy video.
Fail it if it contains anything frightening, violent, sexual, hateful, or political, anything about real people, anything using trademarked characters, or anything a parent would not want a young child to see.

Answer only with a JSON object in this form: {{"ok": true or false, "reason": "short reason"}}

Script:
{json.dumps(script, indent=2)}
"""
    result = ask_claude_json(prompt, max_tokens=300)
    print("Safety review:", result)
    return bool(result.get("ok"))


def narrate(text, out_path):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers=openai_headers(),
        json={"model": "tts-1", "voice": "nova", "input": text, "response_format": "mp3"},
        timeout=120,
    )
    response.raise_for_status()
    out_path.write_bytes(response.content)


def draw_scene(image_prompt, out_path):
    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers=openai_headers(),
        json={
            "model": "gpt-image-1",
            "prompt": image_prompt + " Bright cartoon style. No text, no logos.",
            "size": "1536x1024",
            "quality": "medium",
        },
        timeout=300,
    )
    response.raise_for_status()
    out_path.write_bytes(base64.b64decode(response.json()["data"][0]["b64_json"]))


def make_clip(image, audio, out):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(image),
            "-i", str(audio),
            "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p",
            "-c:v", "libx264", "-tune", "stillimage", "-r", "25",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(out),
        ],
        check=True,
    )


def join_clips(clips, out):
    (WORK / "clips.txt").write_text("".join(f"file '{clip.name}'\n" for clip in clips))
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "clips.txt",
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(out),
        ],
        cwd=WORK,
        check=True,
    )


def upload_to_youtube(video_path, script):
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": script["title"][:100],
            "description": script["description"],
            "tags": script["tags"],
            "categoryId": "23",  # 23 = Comedy
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": True,
            "containsSyntheticMedia": True,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    return response["id"]


def main():
    if PAUSE_FILE.exists():
        print("Paused. Delete the PAUSED file to resume.")
        return

    WORK.mkdir(exist_ok=True)
    history = load_history()

    script = make_script(history)
    print("Today's title:", script["title"])
    if not passes_safety_review(script):
        raise SystemExit("Safety review failed, so nothing was uploaded.")

    clips = []
    for i, scene in enumerate(script["scenes"][:MAX_SCENES]):
        image = WORK / f"scene{i}.png"
        audio = WORK / f"scene{i}.mp3"
        clip = WORK / f"clip{i}.mp4"
        draw_scene(scene["image_prompt"], image)
        narrate(scene["narration"], audio)
        make_clip(image, audio, clip)
        clips.append(clip)
        print(f"Scene {i + 1} done")

    final = (WORK / "final.mp4").resolve()
    join_clips(clips, final)

    video_id = upload_to_youtube(final, script)
    history.append(script["title"])
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    print(f"Published: https://youtu.be/{video_id}")


if __name__ == "__main__":
    main()
