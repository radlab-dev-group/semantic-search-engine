from bs4 import BeautifulSoup


def read_all_entities_by_name(html_content: str, entity_name: str) -> list:
    entities = []
    try:
        soup = BeautifulSoup(html_content, "xml")
        for e in soup.find_all(entity_name):
            entities.append(e)
    except Exception:
        return []
    return entities


# pl_wn_path = "/mnt/data2/data/lexicons/plwordnet-4.2/plwordnet_4_2/plwordnet_4_2__sample.xml"
pl_wn_path = "/mnt/data2/data/lexicons/plwordnet-4.2/plwordnet_4_2/plwordnet_4_2.xml"

with open(pl_wn_path, "r") as pl_wn_file:
    pl_wn_xml_str = pl_wn_file.read()
    pl_wn_all_lu = read_all_entities_by_name(
        pl_wn_xml_str, entity_name="lexical-unit"
    )
    pl_wn_all_synset = read_all_entities_by_name(pl_wn_xml_str, entity_name="synset")
    relation_types = read_all_entities_by_name(
        pl_wn_xml_str, entity_name="relationtypes"
    )
    lexical_relations = read_all_entities_by_name(
        pl_wn_xml_str, entity_name="lexicalrelations"
    )
    synset_relations = read_all_entities_by_name(
        pl_wn_xml_str, entity_name="synsetrelations"
    )

    print(relation_types)


# import spacy
#
# text_str = "Dzisiaj około godziny 13 straż pożarna dostała wezwanie do jednego z domostw we wsi Bystre (gmina Oleśnica). Mieszkańcy zauważyli dym wydobywający się z gniazdek elektrycznych. Służby są już na miejscu i sprawdzają obiekt."
#
# nlp_spacy = spacy.load("pl_core_news_lg")
#
# doc = nlp_spacy(text_str)
#
# for s in doc.sents:
#     sentence = s.text.strip()
#     print(dir(s))
#     print(s.lemma_)
#     print(sentence)
#     for token in s.subtree:
#         print("\t", token.text, token.lemma_, token.tag_, token.ent_type_, token.ent_id_)
#
