#
# Based on the code from the ANCE repository
# (MIT license, which is compatible with the current repo's Apache license):
#
# https://github.com/microsoft/ANCE/blob/master/data/msmarco_data.py
# https://github.com/microsoft/ANCE/blob/master/data/DPR_data.py
#
# This code could have had much less copy-paste, but we try to
# keep as close to the original as possible.
#
import json

from flexneuart.config import DOCID_FIELD, TEXT_RAW_FIELD_NAME
from flexneuart.text_proc.clean import replace_tab
from flexneuart.io import FileWrapper, jsonl_gen, multi_file_linegen
from flexneuart.data_convert import MSMARCO_DOC_V2_FILE_PATTERN, MSMARCO_PASS_V2_FILE_PATTERN

MSMARCO_MAX_QUERY_SEQ = 64
MSMARCO_MAX_DOC_SEQ = 512
# keep only first 10000 characters, should be sufficient for any
# experiment that uses less than 500 - 1k tokens
MSMARCO_MAX_DOC_LEN = 10000

DPR_MAX_SEQ_LEN = 256

DATA_TYPE_MSMARCO_DOC_FIRSTP = 'msmarco_doc_firstp'
DATA_TYPE_MSMARCO_DOC_V2_FIRSTP = 'msmarco_doc_v2_firstp'
DATA_TYPE_MSMARCO_PASS = 'msmarco_pass'
DATA_TYPE_MSMARCO_PASS_V2 = 'msmarco_pass_v2'
DATA_TYPE_DPR_NQ = 'dpr_nq'
DATA_TYPE_DPR_TRIVIA = 'dpr_trivia'

# Must match download_ance_models.sh
DATA_TYPE_PATHS = {
    DATA_TYPE_MSMARCO_DOC_FIRSTP      : 'Document_ANCE_FirstP_Checkpoint',
    DATA_TYPE_MSMARCO_DOC_V2_FIRSTP   : 'Document_ANCE_FirstP_Checkpoint',
    DATA_TYPE_MSMARCO_PASS            : 'Passage_ANCE_FirstP_Checkpoint',
    DATA_TYPE_MSMARCO_PASS_V2         : 'Passage_ANCE_FirstP_Checkpoint',
    DATA_TYPE_DPR_NQ                  : 'nq.cp',
    DATA_TYPE_DPR_TRIVIA              : 'trivia.cp'
}

DATA_TYPE_CHOICES = [DATA_TYPE_DPR_NQ,
                     DATA_TYPE_DPR_TRIVIA,
                     DATA_TYPE_MSMARCO_DOC_FIRSTP,
                     DATA_TYPE_MSMARCO_DOC_V2_FIRSTP,
                     DATA_TYPE_MSMARCO_PASS_V2,
                     DATA_TYPE_MSMARCO_PASS]


def attention_mask(token_seq, max_seq_length):
    assert max_seq_length >= len(token_seq)
    return [1] *len(token_seq) + [0] * (max_seq_length - len(token_seq))


def pad_input_ids_msmarco(input_ids, max_length,
                          pad_on_left=False,
                          pad_token=0):
    """An original padding function that is used only for MS MARCO data."""
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids


def pad_input_ids_dpr(max_seq_length, token_ids, tokenizer):
    """An original padding function that is used only for DPR data."""
    if len(token_ids) < max_seq_length:
        pad_token_ids = token_ids + [tokenizer.pad_token_id] * (max_seq_length - len(token_ids))
    if len(token_ids) >= max_seq_length:
        pad_token_ids = token_ids[0:max_seq_length]
        #pad_token_ids[-1] = tokenizer.sep_token_id # Leo's comment why do they do it here? It can't really make a difference
        # the truncation of DPR data should happen very rarely if at all
    return pad_token_ids


def tokenize_query_msmarco(query_text, tokenizer, max_query_len=MSMARCO_MAX_QUERY_SEQ):
    """Encode MS MARCO query text (without query ID).

    :param query_text: a raw query text (no ID)
    :param tokenizer: a tokenizer
    :param max_query_len: the maximum # of tokens to keep.
    :return: a tuple: (a list of padded token IDs, attention mask)
    """
    query_text = query_text.lower()
    token_ids = tokenizer.encode(query_text.rstrip(),
                                add_special_tokens=True,
                                max_length=max_query_len,
                                truncation=True,
                                padding=False)

    pad_token_ids = pad_input_ids_msmarco(token_ids, max_query_len)
    return pad_token_ids, attention_mask(token_ids, max_query_len)


def tokenize_query_dpr(query_text, tokenizer, max_seq_length=DPR_MAX_SEQ_LEN):
    """Encode DPR query text (without query ID). We keep the original ANCE
       code as much as possible and the way the pad queries for DPR is not the
       same as they do it for MS MARCO. However, the way they compute the attention
       mask seems to be wrong, so we keep the same procedure as for MS MACO.

    :param query_text: a raw query text (no ID)
    :param tokenizer: a tokenizer
    :param max_seq_length: the maximum # of tokens to keep.
    :return: a tuple: (a list of padded token IDs, attention mask)
    """
    token_ids = tokenizer.encode(query_text,
                                 add_special_tokens=True,
                                 max_length=max_seq_length,
                                 truncation=True,
                                 padding=False)

    pad_token_ids = pad_input_ids_dpr(max_seq_length, token_ids, tokenizer)

    return pad_token_ids, attention_mask(token_ids, max_seq_length)


def parse_and_tokenize_msmarco(is_doc, line, tokenizer,
                               max_doc_character=MSMARCO_MAX_DOC_LEN,
                               max_seq_length=MSMARCO_MAX_DOC_SEQ):
    """Parse a source line from he MS MARCO document or passage file an tokenize it.

    :param is_doc: True for documents and False for passages
    :param line: a raw input line
    :param tokenizer: a tokenizer object
    :param max_doc_character: the max. # of characters to keep
    :param max_seq_length: the maximum length of the target sequence
    :return: a triple: (document or passage id, a list of padded token IDs, attention mask)
    """

    if is_doc:
        line_arr = line.split('\t')
        assert len(line_arr) == 4, f'Improper format MS MARCO documents, line: {line}'
        doc_id = line_arr[0] # no lowercasing of document IDs

        url = line_arr[1].rstrip().lower()
        title = line_arr[2].rstrip().lower()
        p_text = line_arr[3].rstrip().lower()

        # <sep> does not seem to be a proper token, this is likely a mistake in ANCE code.
        # However, we keep it as the encoders were trained with this mistake
        full_text = url + "<sep>" + title + "<sep>" + p_text
        full_text = full_text[:max_doc_character]
    else:
        line = line.strip()
        line_arr = line.split('\t')
        assert len(line_arr) == 2, f'Improper format MS MARCO passages, line: {line}'
        doc_id = line_arr[0] # no lowercasing of document IDs

        p_text = line_arr[1].rstrip().lower()

        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = p_text[:max_doc_character]

    token_ids = tokenizer.encode(full_text,
                                 add_special_tokens=True,
                                 max_length=max_seq_length,
                                 truncation=True,
                                 padding=False)
    pad_token_ids = pad_input_ids_msmarco(token_ids, max_seq_length)

    return doc_id, pad_token_ids, attention_mask(token_ids, max_seq_length)


def parse_and_tokenize_dpr(line, tokenizer, max_seq_length=DPR_MAX_SEQ_LEN):
    """Parse a source line from the DPR passage file.

    :param line: a raw input line
    :param tokenizer: a tokenizer object
    :param max_seq_length: the maximum length of the target sequence
    :return: a triple: (passage id, a list of padded token IDs, attention mask)
    """
    line_arr = line.split('\t')

    assert len(line_arr) == 3, f'Improper format Wikipedia DPR passages, line: {line}'

    pass_id = line_arr[0]  # no lowercasing of passage ID
    text = line_arr[1].lower()
    title = line_arr[2].lower()

    token_ids = tokenizer.encode(title, text_pair=text,
                                 add_special_tokens=True,
                                 max_length=max_seq_length,
                                 truncation=True,
                                 padding=False)

    pad_token_ids = pad_input_ids_dpr(max_seq_length, token_ids, tokenizer)

    return pass_id, pad_token_ids, attention_mask(token_ids, max_seq_length)


# To unify processing we provide a few helper "generator" function
def wikipedia_dpr_body_generator(input_file, tokenizer):
    """DPR generator.

    :param input_file: input file name
    :param tokenizer: a tokenizer object
    :return: yields a triple: (passage id, attention mask, a list of padded token IDs)
    """
    first = True
    with FileWrapper(input_file) as inpf:
        for line in inpf:
            # Skip the first line with IDs
            if first:
                first = False
                continue

            yield parse_and_tokenize_dpr(line, tokenizer)


# To unify processing we provide a few helper "generator" function
def msmarco_body_generator(input_file, is_doc, tokenizer):
    """MS MARCO generator.

    :param input_file: input file name
    :param is_doc: True for documents and False for passages
    :param tokenizer: a tokenizer object
    :return: yields a triple: (passage/document id, attention mask, a list of padded token IDs)
    """
    with FileWrapper(input_file) as inpf:
        for line in inpf:
            yield parse_and_tokenize_msmarco(is_doc, line, tokenizer)


def msmarco_doc_v2_body_generator(input_dir, tokenizer):
    """MS MARCO (v2) document generator.

    :param input_dir:   an input directory with un-tarred (compressed) JSONL files
    :param tokenizer:   tokenizer: a tokenizer object
    :return:  yields a triple: (passage/document id, attention mask, a list of padded token IDs)
    """

    for line in multi_file_linegen(input_dir, MSMARCO_DOC_V2_FILE_PATTERN):
        fields = json.loads(line)
        body = replace_tab(fields['body'])
        did = replace_tab(fields['docid'])
        title = replace_tab(fields['title'])
        url = replace_tab(fields['url'])
        yield parse_and_tokenize_msmarco(is_doc=True,
                                         line='\t'.join([did, url, title, body]),
                                         tokenizer=tokenizer)


def msmarco_pass_v2_body_generator(input_dir, tokenizer):
    """MS MARCO (v2) passage generator.

    :param input_dir:   an input directory with un-tarred (compressed) JSONL files
    :param tokenizer:   tokenizer: a tokenizer object
    :return:  yields a triple: (passage/document id, attention mask, a list of padded token IDs)
    """

    for line in multi_file_linegen(input_dir, MSMARCO_PASS_V2_FILE_PATTERN):
        fields = json.loads(line)
        passage = replace_tab(fields['passage'])
        did = replace_tab(fields['pid'])
        yield parse_and_tokenize_msmarco(is_doc=False,
                                         line='\t'.join([did, passage]),
                                         tokenizer=tokenizer)


def jsonl_query_generator(input_file, tokenizer, tokenize_func):
    for data_entry in jsonl_gen(input_file):
        doc_id = data_entry[DOCID_FIELD]
        pad_token_ids, attention_mask = tokenize_func(data_entry[TEXT_RAW_FIELD_NAME], tokenizer)
        yield doc_id, pad_token_ids, attention_mask


