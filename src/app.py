import streamlit as st
from rag import ask



st.title("Document Assistant")
question=st.text_input("Ask a question about the ZATCA VAT documents")


if question:
    with st.spinner("Searching and writing the answer..."):
        answer, sources ,used_chunks= ask(question)
    st.write(answer)
    if sources:
        st.caption("Sources: " + ", ".join(sources))
    if used_chunks:
        with st.expander("Show the text the answer was based on"):
            for c in used_chunks:
                st.markdown(f"**{c['source']}**")
                st.text(c["text"])
                st.divider()

