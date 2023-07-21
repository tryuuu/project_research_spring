import openai
openai.api_key = "sk-"
while True:
  user_input = input("あなた: ")
  if user_input == "終了":
    break
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=[
      #{"role": "system", "content":"You are Micky Mouse. Pretend you are him. Please respond briefly."},
      {"role": "system", "content":f"今から、しりとりをします。受け取った言葉をひらがなにした時最後の文字から始まる言葉のみを言ってください。最後が「ん」で終わらないよう注意してください"},
      {"role": "user", "content": f"{user_input}"}])
  response = response['choices'][0]['message']['content']
  print(f"GPT: {response}")
