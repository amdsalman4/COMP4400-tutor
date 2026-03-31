"""
app.py — Gradio chat interface for the COMP-4400 RAG tutor.

Usage:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from tutor import Tutor

bot = Tutor()


def respond(user_message: str, chat_history: list[list[str | None]]):
    """Called by Gradio on each user submission."""
    if not user_message.strip():
        return chat_history, "", ""

    answer, sources = bot.ask(user_message)

    # Format sources as a readable string for the sidebar panel
    sources_text = "\n".join(f"• {s}" for s in sources)

    chat_history.append([user_message, answer])
    return chat_history, "", sources_text


def clear_chat():
    bot.reset()
    return [], "", "No sources yet."


# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="COMP-4400 Tutor") as demo:
    gr.Markdown(
        """
        # COMP-4400 Tutor
        **Principles of Programming Languages — University of Windsor**

        Ask about lambda calculus, Scheme, Prolog, MapReduce, axiomatic semantics,
        code optimization, garbage collection, AOP, LLMs, and more.
        Answers are grounded in your course materials.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about COMP-4400…",
                    show_label=False,
                    scale=5,
                    submit_btn=True,
                )
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Retrieved Sources")
            sources_box = gr.Textbox(
                value="No sources yet.",
                label="",
                lines=18,
                interactive=False,
            )

    # Wire up events
    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, sources_box],
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_box, sources_box],
    )

    gr.Markdown(
        "_Powered by OpenAI gpt-4o-mini · sentence-transformers/all-MiniLM-L6-v2 · ChromaDB_"
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
