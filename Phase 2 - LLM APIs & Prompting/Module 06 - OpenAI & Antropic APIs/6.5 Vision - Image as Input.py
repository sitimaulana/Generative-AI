import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

def describe_image_url(url: str) -> str:
    response = client.chat.completions.create(
        # Gunakan model yang terbukti mendukung Vision
        model="openai/gpt-4o",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe what you see in this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    }
                }
            ]
        }]
    )

    return response.choices[0].message.content


# Penggunaan dengan Direct Image URL (contoh gambar Bunga Matahari dari Wikimedia)
image_url = "https://upload.wikimedia.org/wikipedia/commons/4/40/Sunflower_sky_backdrop.jpg"

print("Sedang menganalisis gambar dari URL...")
text = describe_image_url(image_url)
print(text)