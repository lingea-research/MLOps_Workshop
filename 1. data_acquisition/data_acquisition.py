from selenium import webdriver
from selenium.webdriver.common.by import By


def extract_language():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=options)

    language_codes = ['tur', 'eng', 'ces', 'slk', 'eng']
    for lang in language_codes:
        with open(f"{lang}.txt", 'w', encoding="utf-8") as file:
            final_sentence_list = []

            for page_num in range(1,100):
                driver.get(f"https://tatoeba.org/en/sentences/show_all_in/{lang}/none?page={page_num}")
                sentences = driver.find_elements(By.TAG_NAME, 'span')
                for _ in set(sentences):
                    if "Sentence" not in _.text and _.text not in final_sentence_list and _.text != '':
                        final_sentence_list.append(f"__label__{lang} "+_.text)

            to_txt = '\n'.join(set(final_sentence_list))
            file.write(to_txt)
    driver.quit()


if __name__ == "__main__":
    extract_language()

