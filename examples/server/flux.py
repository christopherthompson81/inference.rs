from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

result = client.images.generate(
    model="default",
    prompt="A vibrant sunset in the mountains, 4k, high quality.",
    n=1,
)
# url is a path on the server, /v1/files/<id>/content, served from its file store for 30 minutes.
print(f"http://localhost:1234{result.data[0].url}")
