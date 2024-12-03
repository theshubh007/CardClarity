import streamlit as st
from Rag_Assistant import CreditCardAssistant


class CardClarityApp:
    def __init__(self):
        self.assistant = CreditCardAssistant()

    def run(self):
        st.title("CardClarity: Credit Card Assistant")
        self.setup_sidebar()
        self.main_content()

    def setup_sidebar(self):
        st.sidebar.title("Enter CreditCard MainPage URLs")
        urls = []
        for i in range(2):
            url = st.sidebar.text_input(f"URL {i+1}")
            urls.append(url)
        process_url_clicked = st.sidebar.button("Process URLs")
        if process_url_clicked:
            self.assistant.process_urls(urls)

    def main_content(self):
        main_placeholder = st.empty()
        query = main_placeholder.text_input(
            "What do you need help with regarding your credit card?"
        )
        if query:
            answer, sources = self.assistant.process_query(query)
            st.header("Answer")
            st.write(answer)
            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(source)


if __name__ == "__main__":
    app = CardClarityApp()
    app.run()
