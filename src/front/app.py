import streamlit as st

from pages import TestImgPage, RandomImgPage, SelectImgPage


def main() -> None:
    backend_address = 'localhost:10000'
    pages = {
        'Test': TestImgPage(title='Test image', backend=backend_address),
        'Random images': RandomImgPage(title='Random images', backend=backend_address),
        'Select images': SelectImgPage(title='Select images', backend=backend_address)
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Sections", list(pages.keys()))
    selected_page = pages[selection]

    with st.spinner(f"Loading {selection} ..."):
        selected_page.write()


if __name__ == '__main__':
    main()
