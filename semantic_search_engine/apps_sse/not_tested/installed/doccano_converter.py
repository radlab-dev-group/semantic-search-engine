"""
1. How to prepare dataset for sequence classification (from doccano export):

    python doccano_converter.py \
        -I /mnt/data/radlab-data/radlab-projekty/ING-Prostomat2/ton-www/doccano/20231208 \
        -e .jsonl \
        --show-class-labels-histogram \
        -O prepared_datasets/ton_of_web/20231208 \
        --token-classification


2. How to prepare dataset for token classification (from doccano export):

    python doccano_converter.py \
        -I /mnt/data/radlab-data/radlab-projekty/ING-Prostomat2/etap2/20231208 \
        -e .jsonl \
        --show-class-labels-histogram \
        -O prepared_datasets/e2/20231208 \
        --sequence-classification
"""

from radlab_data.preprocessing.dataset import (
    SaveConfiguration,
    SequenceLabellingDataset,
    TextClassificationDataset,
)
from radlab_data.preprocessing.pipeline import Pipeline
from radlab_data.preprocessing.pipeline_modules import (
    AlignAnnotationToWordBoundaries,
    RemoveDuplicates,
    RemoveLongIOBAnnotation,
    RemoveUnlabelledData,
    SplitSentences,
)
from radlab_data.utils.argument_parser import (
    INPUT_DIR_REQUIRED,
    OUTPUT_DIR_REQUIRED,
    prepare_parser_for_fields,
)


def prepare_parser(desc=""):
    p = prepare_parser_for_fields(
        fields_list=[INPUT_DIR_REQUIRED, OUTPUT_DIR_REQUIRED], description=desc
    )
    p.add_argument(
        "-m", "--mapping-file", dest="mapping_file", type=str, required=False
    )
    p.add_argument(
        "-e",
        "--dataset-extension",
        dest="dataset_extension",
        type=str,
        default=".jsonl",
        required=False,
    )
    p.add_argument(
        "--save-excel-annotations",
        dest="save_excel_annotations",
        action="store_true",
    )
    p.add_argument(
        "--save-iob-standard", dest="save_iob_standard", action="store_true"
    )
    p.add_argument(
        "--show-class-labels-histogram",
        dest="show_class_labels_histogram",
        action="store_true",
    )

    p.add_argument(
        "--sequence-classification",
        dest="sequence_classification",
        action="store_true",
    )
    p.add_argument(
        "--token-classification",
        dest="token_classification",
        action="store_true",
    )

    return p


def prepare_pipeline(is_sequence_classification: bool = False):
    # pipeline_modules = [RemoveUnlabelledData(), SplitSentences(), RemoveDuplicates()]
    # pipeline_modules = [SplitSentences(), RemoveDuplicates()]
    pipeline_modules = []
    if not is_sequence_classification:
        pipeline_modules.extend(
            [
                RemoveUnlabelledData(),
                # RemoveLongIOBAnnotation(ans_factor=0.55),
                AlignAnnotationToWordBoundaries(align_begin=True, align_end=True),
            ]
        )
    pipeline_modules.append(RemoveUnlabelledData())
    return Pipeline(modules=pipeline_modules)


def main(argv=None):
    args = prepare_parser().parse_args(args=argv)

    if (not args.sequence_classification and not args.token_classification) or (
        args.sequence_classification and args.token_classification
    ):
        raise Exception(
            "Choose one of the mode: --sequence-classification or --token-classification"
        )
    pipeline = prepare_pipeline(args.sequence_classification)

    _dataset_cls = (
        TextClassificationDataset
        if args.sequence_classification
        else SequenceLabellingDataset
    )

    pre_dataset = _dataset_cls(
        dataset_path=args.input_dir,
        dataset_extension=args.dataset_extension,
        class_label_field_name="label",
        class_labels_mapping_file=args.mapping_file,
    )

    pre_dataset = pipeline(pre_dataset)

    save_config = SaveConfiguration(
        path=args.output_dir,
        add_date_created_to_name=True,
        save_iob_file=args.save_iob_standard,
        save_text_label_xlsx=args.save_excel_annotations,
        iob_window_o_class=4,
        show_class_labels_histogram=args.show_class_labels_histogram,
    )

    pre_dataset.save(save_config)


if __name__ == "__main__":
    main()
