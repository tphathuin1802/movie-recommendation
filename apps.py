import streamlit as st

st.title("Movies Recommender System")

st.write(
    "This is a simple movie recommender system that recommends movies based on the user's input."
)
st.write("Author: Huynh Tan Phat")
st.link_button("Visit my website", "https://stephen-huynh.vercel.app/")

selected_movie = st.selectbox(
    "Tap to select your favorite movie", ["Email", "Home phone", "Mobile phone"]
)

if st.button("Show Recommendation"):
    st.write(f"You selected: {selected_movie}")
