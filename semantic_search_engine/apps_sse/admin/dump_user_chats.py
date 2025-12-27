import os
import json
import django
import argparse
import pandas as pd

from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()


from data.models import OrganisationUser
from chat.models import Chat, Message


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("-o", "--out-dir", dest="out_dir", required=True)
    return p


def get_all_users_from_db():
    return OrganisationUser.objects.all()


def get_all_user_chats(user: OrganisationUser):
    return Chat.objects.filter(organisation_user=user).order_by("created_at")


def prepare_user_name_as_file_name(user: OrganisationUser):
    return user.auth_user.username.replace("@", "___").replace(" ", "_").strip()


def get_chat_messages(chat: Chat):
    return Message.objects.filter(chat=chat).order_by("number")


def prepare_chat_as_dict_with_rating(chat: Chat, empty_val: str = "-") -> {}:
    chat_messages = get_chat_messages(chat=chat)
    if not len(chat_messages):
        return {}

    chat_history = []
    for message in chat_messages:
        msg_as_dict = {
            "datetime": message.date_time.strftime("%Y-%m-%dT %H:%M:%S"),
            "user": empty_val,
            "assistant": empty_val,
            "rate_value": -1,
            "rate_value_max": -1,
            "rate_comment": empty_val,
            "templates": empty_val,
            "sse_query": empty_val,
            "detailed_results": empty_val,
            "general_results": empty_val,
            "structured_results": empty_val,
        }

        # User/assistant/other-role message
        if message.role == "user":
            msg_as_dict["user"] = message.text
        elif message.role == "assistant":
            msg_as_dict["assistant"] = message.text
        else:
            msg_as_dict[message.role] = message.text

        # Put state info
        if message.state and message.state.rag_message_state:
            r_state = message.state.rag_message_state
            if r_state.sse_query:
                # sse query str
                msg_as_dict["sse_query"] = r_state.sse_query.query_str_prompt

                # sse templates names
                templ_str = ""
                for t in r_state.sse_query.query_templates.all():
                    if len(templ_str):
                        templ_str += ", "
                    templ_str += t.name
                msg_as_dict["templates"] = templ_str

            if r_state.sse_response:
                msg_as_dict["general_results"] = (
                    r_state.sse_response.general_stats_json
                )
                msg_as_dict["detailed_results"] = (
                    r_state.sse_response.detailed_results_json
                )
                msg_as_dict["structured_results"] = (
                    r_state.sse_response.structured_results
                )

            if r_state.sse_answer:
                if r_state.sse_answer.rate_value:
                    msg_as_dict["rate_value"] = r_state.sse_answer.rate_value
                if r_state.sse_answer.rate_nax_value:
                    msg_as_dict["rate_value_max"] = r_state.sse_answer.rate_nax_value
                if r_state.sse_answer.rate_comment:
                    msg_as_dict["rate_comment"] = r_state.sse_answer.rate_comment
        chat_history.append(msg_as_dict)

    return {
        "id": chat.pk,
        "hash": chat.hash,
        "created": chat.created_at.strftime("%Y-%m-%dT %H:%M:%S"),
        "history": chat_history,
    }


def prepare_user_full_chat_dump(out_dir: str, user: OrganisationUser):
    base_out_dir = os.path.join(out_dir, user.organisation.name)
    os.makedirs(base_out_dir, exist_ok=True)
    out_user_file_no_ext = os.path.join(
        base_out_dir, prepare_user_name_as_file_name(user=user)
    )

    all_user_chats_as_dicts = []
    all_user_chats_objects = get_all_user_chats(user=user)
    for chat in all_user_chats_objects:
        chat_as_dict = prepare_chat_as_dict_with_rating(chat)
        if not len(chat_as_dict):
            continue
        all_user_chats_as_dicts.append(chat_as_dict)

    out_json = out_user_file_no_ext + ".json"
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(all_user_chats_as_dicts, indent=2, ensure_ascii=False))

    out_xlsx = out_user_file_no_ext + ".xlsx"
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        for chat_as_dict in all_user_chats_as_dicts:
            sheet_name = str(chat_as_dict["id"])
            sheet_df = pd.DataFrame(chat_as_dict["history"])
            sheet_df.to_excel(writer, sheet_name, index=False)


def main(argv=None):
    args = prepare_parser().parse_args()
    all_org_users = get_all_users_from_db()

    with tqdm(total=len(all_org_users), desc="Preparing dump of ratings") as pbar:
        for o_user in all_org_users:
            prepare_user_full_chat_dump(out_dir=args.out_dir, user=o_user)
            pbar.update(1)


if __name__ == "__main__":
    main()
