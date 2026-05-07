import gradio as gr
import time

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

if __name__ == "__main__":
    print("🚀 Starting test server...")
    demo.launch(server_name="0.0.0.0", server_port=7861)
    time.sleep(5)
    print("✅ Test server started (and would continue, but we exit for test)")
