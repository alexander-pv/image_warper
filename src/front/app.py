import streamlit as st

import pages as st_page


def main() -> None:
    pages_dict = {
        'Test': st_page.TestImgPage(title='Test image'),
        'Random pair': st_page.RandomImgPage(title='Random images'),
        'Select pair': st_page.SelectImgPage(title='Select images'),
        'Test three images': st_page.TestThreeImgPage(title='Test three images input'),
        'Select three images': st_page.SelectThreeImgPage(title='Test three images input'),
        'Style transfer': st_page.StyleTransferPage(title='Test three images input')
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Sections", list(pages_dict.keys()))
    selected_page = pages_dict[selection]

    with st.spinner(f"Loading {selection} ..."):
        selected_page.write()


if __name__ == '__main__':
    main()
