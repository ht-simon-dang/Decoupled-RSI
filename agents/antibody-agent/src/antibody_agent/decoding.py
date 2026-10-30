"""Transformers 5 bridge using LM Format Enforcer's public core API.

The upstream transformers integration in 0.11.3 imports a removed Transformers 4
symbol. This adapter avoids modifying either installed dependency.
Tokenizer boundary handling follows the documented upstream integration:
https://github.com/noamgat/lm-format-enforcer (MIT).
"""

from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


def tokenizer_data(tokenizer):
    special = set(tokenizer.all_special_ids)
    ids = sorted(i for i in tokenizer.get_vocab().values() if i not in special)
    prefix = tokenizer.encode("0", add_special_tokens=False)[-1]
    plain = tokenizer.batch_decode([[i] for i in ids], clean_up_tokenization_spaces=False)
    following = tokenizer.batch_decode([[prefix, i] for i in ids], clean_up_tokenization_spaces=False)
    regular = [
        (i, after[1:], len(after[1:]) > len(before))
        for i, before, after in zip(ids, plain, following, strict=True)
    ]

    def decode(tokens):
        return tokenizer.decode(tokens, clean_up_tokenization_spaces=False).rstrip("\ufffd")

    return TokenEnforcerTokenizerData(regular, decode, tokenizer.eos_token_id, False, len(tokenizer))


def prefix_function(data, schema):
    enforcer = TokenEnforcer(data, JsonSchemaParser(grammar_schema(schema)))

    def allowed(batch_id, sent):
        choices = enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
        # Once the object is complete, stop instead of spending the budget on trailing whitespace.
        if data.eos_token_id in choices:
            return [data.eos_token_id]
        return choices

    return allowed


def grammar_schema(value):
    """LMFE 0.11 cannot combine regex with length; full bounds remain in Pydantic."""
    if isinstance(value, list):
        return [grammar_schema(item) for item in value]
    if isinstance(value, dict):
        return {
            key: grammar_schema(item)
            for key, item in value.items()
            if not ("pattern" in value and key in {"minLength", "maxLength"})
        }
    return value
