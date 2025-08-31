import collections
import copy
import json
import pathlib
import typing
import warnings

from beancount_black.formatter import parse_date
from beancount_parser.data_types import Entry
from beancount_parser.data_types import EntryType
from beancount_parser.helpers import collect_entries
from beancount_parser.parser import make_parser
from beancount_parser.parser import traverse
from lark import Lark
from lark import Token
from lark import Tree

from . import constants
from .data_types import Amount
from .data_types import BeancountTransaction
from .data_types import ChangeSet
from .data_types import DeletedTransaction
from .data_types import GeneratedBalance
from .data_types import GeneratedPosting
from .data_types import GeneratedTransaction
from .data_types import ImportOverrideFlag
from .data_types import MetadataItem
from .data_types import TransactionStatement
from .data_types import TransactionUpdate


def parse_override_flags(value: str) -> frozenset[ImportOverrideFlag] | None:
    parts = value.split(",")
    try:
        flags = frozenset(map(ImportOverrideFlag, parts))
    except ValueError:
        warnings.warn(f"Invalid override flags: {value}", RuntimeWarning)
        return
    if (ImportOverrideFlag.ALL in flags or ImportOverrideFlag.NONE in flags) and len(
        flags
    ) > 1:
        warnings.warn(
            f"When NONE or ALL present in the override flags, there should be no other flags but we got {value}",
            RuntimeWarning,
        )
        return
    return flags


def extract_existing_transactions(
    parser: Lark,
    bean_file: pathlib.Path,
    root_dir: pathlib.Path | None = None,
) -> typing.Generator[BeancountTransaction, None, None]:
    for bean_path, tree in traverse(
        parser=parser, bean_file=bean_file, root_dir=root_dir
    ):
        last_txn = None
        import_id = None
        import_override = None
        if tree.data != "start":
            raise ValueError("Expected start")
        for child in tree.children:
            if child is None:
                continue
            if child.data != "statement":
                raise ValueError("Expected statement")
            first_child = child.children[0]
            if not isinstance(first_child, Tree):
                continue
            if first_child.data == "date_directive":
                date_directive = first_child.children[0]
                directive_type = date_directive.data.value
                if directive_type not in ("txn", "balance"):
                    continue
                if last_txn is not None and import_id is not None:
                    yield BeancountTransaction(
                        file=bean_path,
                        lineno=last_txn.meta.line,
                        id=import_id,
                        override=import_override,
                    )
                import_id = None
                import_override = None
                last_txn = date_directive
            elif first_child.data == "metadata_item":
                metadata_key = first_child.children[0].value
                metadata_value = first_child.children[1]
                if metadata_value.type == "ESCAPED_STRING":
                    metadata_value_str = json.loads(metadata_value.value)
                    if metadata_key == constants.IMPORT_ID_KEY:
                        import_id = metadata_value_str
                    elif metadata_key == constants.IMPORT_OVERRIDE_KEY:
                        import_override = parse_override_flags(metadata_value_str)
        if last_txn is not None and import_id is not None:
            yield BeancountTransaction(
                file=bean_path,
                lineno=last_txn.meta.line,
                id=import_id,
                override=import_override,
            )


def compute_changes(
    generated_txns: list[GeneratedTransaction],
    imported_txns: list[BeancountTransaction],
    work_dir: pathlib.Path,
    deleted_txns: list[DeletedTransaction] | None = None,
    generated_balances: list[GeneratedBalance] | None = None,
) -> dict[pathlib.Path, ChangeSet]:
    generated_id_txns = {txn.id: txn for txn in generated_txns}
    imported_id_txns = {txn.id: txn for txn in imported_txns}
    deleted_txn_ids = set(txn.id for txn in (deleted_txns or ()))
    
    # Create a combined lookup that includes both transactions and balance assertions
    # This prevents balance assertions from being marked as dangling
    all_generated_ids = set(generated_id_txns.keys())
    if generated_balances:
        all_generated_ids.update(balance.id for balance in generated_balances)

    to_remove = collections.defaultdict(list)
    dangling_txns = collections.defaultdict(list)
    for txn in imported_txns:
        if txn.id in deleted_txn_ids:
            to_remove[txn.file].append(txn)
            continue
        generated_txn = generated_id_txns.get(txn.id)
        if (
            generated_txn is not None
            and txn.file.resolve() != (work_dir / generated_txn.file).resolve()
        ):
            # it appears that the generated txn's file is different from the old one, let's remove it
            to_remove[txn.file].append(txn)
        elif txn.id not in all_generated_ids and txn.override is None:
            # we have existing imported txn without override flags but has no corresponding generated txn or balance,
            # let's add it to danging txns
            dangling_txns[txn.file].append(txn)

    to_add = collections.defaultdict(list)
    to_update = collections.defaultdict(dict)
    for txn in generated_txns:
        if txn.id in deleted_txn_ids:
            continue
        imported_txn = imported_id_txns.get(txn.id)
        generated_file = (work_dir / txn.file).resolve()
        if imported_txn is not None and imported_txn.file.resolve() == generated_file:
            to_update[generated_file][imported_txn.lineno] = TransactionUpdate(
                txn=txn, override=imported_txn.override
            )
        else:
            to_add[generated_file].append(txn)

    # Handle balance assertions the same way as transactions
    if generated_balances:
        for balance in generated_balances:
            if balance.id in deleted_txn_ids:
                continue
            imported_txn = imported_id_txns.get(balance.id)
            
            if imported_txn is not None:
                # Balance assertion already exists - use the same file and update it
                balance_file = imported_txn.file.resolve()
                to_update[balance_file][imported_txn.lineno] = TransactionUpdate(
                    txn=_balance_to_transaction(balance), override=imported_txn.override
                )
            else:
                # New balance assertion - determine which file to use
                # Use the first file from to_add, or create a default file
                if to_add:
                    balance_file = next(iter(to_add.keys()))
                else:
                    balance_file = work_dir / "balances.bean"
                # Convert to transaction for the existing pipeline
                to_add[balance_file].append(_balance_to_transaction(balance))

    all_files = (
        frozenset(to_remove.keys())
        .union(to_add.keys())
        .union(to_update.keys())
        .union(dangling_txns)
    )
    return {
        file_path: ChangeSet(
            remove=to_remove[file_path],
            add=to_add[file_path],
            update=to_update[file_path],
            dangling=dangling_txns[file_path],
        )
        for file_path in all_files
    }


def to_parser_entry(parser: Lark, text: str, lineno: int | None = None) -> Entry:
    tree = parser.parse(text.strip())
    entries, _ = collect_entries(tree)
    if len(entries) != 1:
        raise ValueError("Expected exactly only one entry")
    entry = entries[0]
    if lineno is not None:
        entry.statement.meta.line = lineno
    return entry


def posting_to_text(posting: GeneratedPosting) -> str:
    columns = [
        posting.account,
    ]
    if posting.amount is not None:
        columns.append(f"{posting.amount.number} {posting.amount.currency}")
    if posting.cost is not None:
        columns.append(posting.cost)
    if posting.price is not None:
        columns.append(f"@ {posting.price.number} {posting.price.currency}")
    return (" " * 2) + " ".join(columns)


def txn_to_text(
    txn: GeneratedTransaction,
) -> str:
    # Check if this is a balance assertion (marked with flag="balance")
    if txn.flag == "balance":
        # Extract balance information from metadata
        balance_account = None
        balance_amount = None
        balance_currency = None
        meta = {}
        
        if txn.metadata:
            for item in txn.metadata:
                if item.name == "__balance_account__":
                    balance_account = item.value
                elif item.name == "__balance_amount__":
                    balance_amount = item.value
                elif item.name == "__balance_currency__":
                    balance_currency = item.value
                else:
                    meta[item.name] = item.value
        
        # Reconstruct the GeneratedBalance from the transaction data
        balance = GeneratedBalance(
            sources=txn.sources,
            id=txn.id,
            date=txn.date,
            account=balance_account,
            amount=Amount(number=balance_amount, currency=balance_currency),
            meta=meta if meta else None,
        )
        return render_balance_assertion(balance)
    
    columns = [
        txn.date,
        txn.flag,
        *((json.dumps(txn.payee),) if txn.payee is not None else ()),
        json.dumps(txn.narration),
    ]
    if txn.tags is not None:
        columns.extend(map(lambda t: "#" + t, txn.tags))
    if txn.links is not None:
        columns.extend(map(lambda t: "^" + t, txn.links))
    line = " ".join(columns)
    import_src = None
    if txn.sources is not None:
        import_src = ":".join(txn.sources)
    extra_metadata = []
    if txn.metadata is not None:
        for item in txn.metadata:
            if item.name in frozenset(
                [constants.IMPORT_ID_KEY, constants.IMPORT_SRC_KEY]
            ):
                raise ValueError(
                    f"Metadata item name {item.name} is reserved for beanhub-import usage"
                )
            extra_metadata.append(f"  {item.name}: {json.dumps(item.value)}")

    return "\n".join(
        [
            line,
            f"  {constants.IMPORT_ID_KEY}: {json.dumps(txn.id)}",
            *(
                (f"  {constants.IMPORT_SRC_KEY}: {json.dumps(import_src)}",)
                if import_src is not None
                else ()
            ),
            *extra_metadata,
            *(map(posting_to_text, txn.postings)),
        ]
    )


def extract_txn_statement(tree: Tree) -> TransactionStatement:
    if tree.data != "statement":
        raise ValueError("Expected a statement here")
    date_directive = tree.children[0]
    if date_directive.data != "date_directive":
        raise ValueError("Expected a date_directive here")
    txn = date_directive.children[0]
    if txn.data != "txn":
        raise ValueError("Expected a txn here")
    date: Token
    flag: Token
    payee: Token | None
    narration: Token
    annoations: Tree | None
    date, flag, payee, narration, annotations = txn.children
    if annotations is not None:
        annotation_values = [annotation.value for annotation in annotations.children]
        links = list(filter(lambda v: v.startswith("^"), annotation_values))
        links.sort()
        hashes = list(filter(lambda v: v.startswith("#"), annotation_values))
        hashes.sort()
    else:
        links = None
        hashes = None
    return TransactionStatement(
        date=parse_date(date.value),
        flag=flag.value,
        payee=json.loads(payee.value) if payee is not None else None,
        narration=json.loads(narration.value),
        hashtags=hashes,
        links=links,
    )


def gen_txn_statement(txn_statement: TransactionStatement) -> Tree:
    return Tree(
        Token("RULE", "statement"),
        [
            Tree(
                Token("RULE", "date_directive"),
                [
                    Tree(
                        Token("RULE", "txn"),
                        [
                            Token("DATE", str(txn_statement.date)),
                            Token("FLAG", txn_statement.flag),
                            Token("ESCAPED_STRING", json.dumps(txn_statement.payee))
                            if txn_statement.payee is not None
                            else None,
                            Token(
                                "ESCAPED_STRING", json.dumps(txn_statement.narration)
                            ),
                            gen_annotations(
                                hashtags=txn_statement.hashtags,
                                links=txn_statement.links,
                            ),
                        ],
                    )
                ],
            ),
            None,
        ],
    )


def gen_annotations(hashtags: list[str] | None, links: list[str] | None) -> Tree | None:
    if hashtags is None and links is None:
        return
    return Tree(
        Token("RULE", "annotations"),
        [
            *(
                Token("TAG", hashtag)
                for hashtag in (hashtags if hashtags is not None else ())
            ),
            *(Token("LINK", link) for link in (links if links is not None else ())),
        ],
    )


def update_transaction(
    parser: Lark, entry: Entry, transaction_update: TransactionUpdate, lineno: int
) -> Entry:
    new_entry = to_parser_entry(
        parser, txn_to_text(transaction_update.txn), lineno=lineno
    )
    if (
        transaction_update.override is None
        or ImportOverrideFlag.ALL in transaction_update.override
    ):
        return new_entry
    elif ImportOverrideFlag.NONE in transaction_update.override:
        return entry
    replacement = {}
    if frozenset(
        [
            ImportOverrideFlag.DATE,
            ImportOverrideFlag.FLAG,
            ImportOverrideFlag.PAYEE,
            ImportOverrideFlag.NARRATION,
            ImportOverrideFlag.HASHTAGS,
            ImportOverrideFlag.LINKS,
        ]
    ).intersection(transaction_update.override):
        txn_statement = extract_txn_statement(entry.statement)
        new_txn_statement = extract_txn_statement(new_entry.statement)
        replacement_statement = gen_txn_statement(
            TransactionStatement(
                **{
                    name: getattr(new_txn_statement, name)
                    if flag in transaction_update.override
                    else getattr(txn_statement, name)
                    for flag, name in [
                        (ImportOverrideFlag.DATE, "date"),
                        (ImportOverrideFlag.FLAG, "flag"),
                        (ImportOverrideFlag.PAYEE, "payee"),
                        (ImportOverrideFlag.NARRATION, "narration"),
                        (ImportOverrideFlag.HASHTAGS, "hashtags"),
                        (ImportOverrideFlag.LINKS, "links"),
                    ]
                }
            )
        )
        replacement_statement.meta.line = entry.statement.meta.line
        replacement["statement"] = replacement_statement
    if ImportOverrideFlag.POSTINGS in transaction_update.override:
        replacement["postings"] = new_entry.postings
    return entry._replace(**replacement)


def render_balance_assertion(balance: GeneratedBalance) -> str:
    """Render a GeneratedBalance as a beancount balance assertion."""
    from . import constants
    import json
    
    # Handle import-src the same way as transactions
    import_src = None
    if balance.sources is not None:
        import_src = ":".join(balance.sources)
    
    # Handle custom metadata
    extra_metadata = []
    if balance.meta is not None:
        for key, value in balance.meta.items():
            if key in frozenset([constants.IMPORT_ID_KEY, constants.IMPORT_SRC_KEY]):
                raise ValueError(
                    f"Metadata item name {key} is reserved for beanhub-import usage"
                )
            extra_metadata.append(f"  {key}: {json.dumps(value)}")
    
    # Build the balance assertion with metadata
    lines = [
        f"{balance.date} balance {balance.account}  {balance.amount.number} {balance.amount.currency}",
        f"  {constants.IMPORT_ID_KEY}: {json.dumps(balance.id)}",
        *(
            (f"  {constants.IMPORT_SRC_KEY}: {json.dumps(import_src)}",)
            if import_src is not None
            else ()
        ),
        *extra_metadata,
    ]
    
    return "\n".join(lines)


def balance_to_text(balance: GeneratedBalance) -> str:
    """Convert a GeneratedBalance to text, similar to txn_to_text."""
    return render_balance_assertion(balance)


def _balance_to_transaction(balance: GeneratedBalance) -> GeneratedTransaction:
    """Convert a GeneratedBalance to a GeneratedTransaction for the existing pipeline.
    
    This allows balance assertions to be processed through the same ChangeSet mechanism
    as transactions, ensuring consistent handling of add/update/remove operations.
    """
    # Store balance amount in metadata
    metadata = []
    if balance.meta:
        for key, value in balance.meta.items():
            metadata.append(MetadataItem(name=key, value=value))
    
    # Add balance-specific metadata
    metadata.extend([
        MetadataItem(name="__balance_account__", value=balance.account),
        MetadataItem(name="__balance_amount__", value=balance.amount.number),
        MetadataItem(name="__balance_currency__", value=balance.amount.currency),
    ])
    
    return GeneratedTransaction(
        sources=balance.sources,
        file="balance.bean",  # Default file for balance assertions
        id=balance.id,
        date=balance.date,
        flag="balance",  # Special flag to mark this as a balance assertion
        payee=None,
        narration="Balance assertion",  # Generic narration
        postings=[],  # Empty postings for balance assertion
        tags=None,
        links=None,
        metadata=metadata,
    )


def apply_change_set(
    tree: Lark,
    change_set: ChangeSet,
    remove_dangling: bool = False,
) -> Lark:
    if tree.data != "start":
        raise ValueError("expected start as the root rule")
    parser = make_parser()

    txns_to_remove = change_set.remove
    if remove_dangling and change_set.dangling is not None:
        txns_to_remove += change_set.dangling
    lines_to_remove = [txn.lineno for txn in txns_to_remove]
    line_to_updates = {
        lineno: txn_update for lineno, txn_update in change_set.update.items()
    }
    entries_to_add = [
        # Set a super huge lineno to the new entry statement as beancount-black sorts entries based on (date, lineno).
        # if we simply add without a proper lineno, it will make sorting unstable.
        to_parser_entry(
            parser, txn_to_text(txn), lineno=constants.ADD_ENTRY_LINENO_OFFSET + i
        )
        for i, txn in enumerate(change_set.add)
    ]

    new_tree = copy.deepcopy(tree)
    entries, tail_comments = collect_entries(new_tree)

    tailing_comments_entry: typing.Optional[Entry] = None
    if tail_comments:
        tailing_comments_entry = Entry(
            type=EntryType.COMMENTS,
            comments=tail_comments,
            statement=None,
            metadata=[],
            postings=[],
        )

    new_children = []

    def _expand_entry(entry: Entry):
        new_children.append(entry.statement)
        for metadata in entry.metadata:
            new_children.extend(metadata.comments)
            new_children.append(metadata.statement)
        for posting in entry.postings:
            new_children.extend(posting.comments)
            new_children.append(posting.statement)
            for metadata in posting.metadata:
                new_children.extend(metadata.comments)
                new_children.append(metadata.statement)

    # Expand existing entries and look up update replacements if there's one
    for entry in entries:
        if entry.type == EntryType.COMMENTS:
            new_children.extend(entry.comments)
            continue
        if entry.statement.meta.line in lines_to_remove:
            # We also drop the comments
            continue
        txn_update = line_to_updates.get(entry.statement.meta.line)
        if txn_update is not None:
            actual_entry = update_transaction(
                parser=parser,
                entry=entry,
                transaction_update=txn_update,
                lineno=entry.statement.meta.line,
            )
        else:
            actual_entry = entry
        # use comments from existing entry regardless
        new_children.extend(entry.comments)
        _expand_entry(actual_entry)

    # Add new entries
    for entry in entries_to_add:
        if entry.type == EntryType.COMMENTS:
            new_children.extend(entry.comments)
            continue
        new_children.extend(entry.comments)
        _expand_entry(entry)

    if tailing_comments_entry is not None:
        new_children.extend(tailing_comments_entry.comments)

    new_tree.children = new_children
    return new_tree
