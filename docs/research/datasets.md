# Dataset sources for LFW and CelebA

Research for issue #2. Checked 2026-09-22. Every URL below was hit with `curl` (HEAD or a
zero-length range GET) on that date; sizes and checksums are from that check, not from memory.

> **Corrections, 2026-09-25.** The dataset fetch ([issue 8](https://github.com/ShikhJohari/Ryuk/issues/8#issuecomment-5800625352)) and later tickets overruled parts of this write-up; the body below is kept as researched.
> - The deep-funneled LFW tarball was fetched from the Wayback Machine memento and its MD5 matches the published value.
> - CelebA images in `flwrlabs/celeba` are pre-cropped **PNG** at 178×218, not JPEG, so YuNet re-detects every face.
> - Only the official valid and test splits were fetched (39,829 images, 1,985 identities); they are the validation and test draws. Each draw uses 500 gallery identities and leaves about 485-500 held-out identities, not about 100 ([issue 9](https://github.com/ShikhJohari/Ryuk/issues/9), [issue 10](https://github.com/ShikhJohari/Ryuk/issues/10)).

## Recommendation

**LFW.** Use `sklearn.datasets.fetch_lfw_pairs(subset="10_folds", funneled=True)`. Under the
hood this pulls `lfw-funneled.tgz` and `pairs.txt` from scikit-learn's own figshare mirror, not
from UMass. No account, no gating, checksum verified on every run.

- `lfw-funneled.tgz`: `https://ndownloader.figshare.com/files/5976015`, 243,346,528 bytes, MD5 `1b42dfed7d15c9b2dd63d5e5840c86ad`.
- `pairs.txt`: `https://ndownloader.figshare.com/files/5976006`, 155,335 bytes, MD5 `9f1ba174e4e1c508ff7cdf10ac338a7d`.
- Licence: none formal. The UMass page states LFW performance "should not be used to conclude
  that an algorithm is suitable for any commercial purpose" and stops there; the underlying
  photos were scraped from Yahoo News and individual photographer copyright was never
  transferred. Treat it as research-only by convention, not by contract.

The official UMass host, `vis-www.cs.umass.edu`, does not resolve as of this research (`NXDOMAIN`
from both the sandbox resolver and Google's `8.8.8.8`), so it cannot be the scriptable source
regardless of what the LFW paper cites. See gotchas below.

If the pipeline specifically needs the deep-funneled tarball (torchvision's default alignment),
there is no clean scriptable mirror. The two options that exist both have friction: the Kaggle
mirror `jessicali9530/lfw-dataset` ships `lfw-deepfunneled.zip` behind Kaggle API auth, and the
Wayback Machine still serves the 2023 UMass copy directly but that's an archival fallback, not a
dependency to script against long-term. Given that, prefer the funneled tarball above and treat
"deep-funneled" as a nice-to-have rather than a requirement.

**CelebA.** Use the Hugging Face dataset `flwrlabs/celeba`, config `img_align+identity+attr`.
Ungated, loads with `datasets.load_dataset("flwrlabs/celeba")`, no Google Drive, no Kaggle
account.

- Repo: `https://huggingface.co/datasets/flwrlabs/celeba`, revision `2d738f56e0e7f925ea36ae7c808ea925264aacec`.
- Splits: train 162,770 / valid 19,867 / test 19,962 rows, summing to the official 202,599 images
  across 10,177 identities.
- Licence: tagged `license:other`, `license_name: celeba-dataset-release-agreement`, with a
  `LICENSE` file in the repo carrying the verbatim CUHK MMLab text (quoted below). Non-commercial
  research only, no redistribution beyond internal use at one organization.
- Size: 25 parquet shards, ~11.7 GB total (exact byte counts in the evidence section).
- Per-file integrity: Hugging Face doesn't publish a single dataset checksum; each parquet blob's
  hash is in the repo's `.gitattributes`/LFS metadata, resolvable via
  `GET https://huggingface.co/api/datasets/flwrlabs/celeba?blobs=true`.

Suggested subset for identification experiments on a laptop: enroll 500 identities (out of
10,177), up to 20 images each, for a watchlist pool of roughly 9,000-10,000 images. Add a
held-out pool of about 100 further identities that are never enrolled, 5-10 images each
(~750 images), to exercise the open-set "nobody on the watchlist" case that Ryuk's own glossary
calls out. Reasoning: CelebA averages 19.9 images per identity but the distribution has a long
tail down to a single image, so 20 as a cap (not a fixed count) uses what's actually available
without inflating the popular identities; identities with fewer than about 5 images aren't worth
enrolling since there's nothing left for a probe split once one or two photos go to enrollment.
At CPU-only embedding speeds, 10,000 faces is minutes of work per recognition model rather than
the hours a full 202,599-image pass would take, and it stays well under a gigabyte on disk against
a 10,177-identity, 1.45 GB+ full download.

## Evidence

### LFW: official UMass site

The canonical page is `https://vis-www.cs.umass.edu/lfw/`. As of 2026-09-22 the hostname does not
resolve: `dig vis-www.cs.umass.edu @8.8.8.8` returns `NXDOMAIN`, answered authoritatively by
`cs.umass.edu`'s own SOA record, and the same failure reproduces from this environment's default
resolver. `cs.umass.edu` itself resolves fine, so this is specific to the `vis-www` subdomain, not
a general UMass outage. Whatever the state was when this issue was filed, it's not scriptable
today.

A Wayback Machine capture from 2025-01-05 (`https://web.archive.org/web/20250105084504/https://vis-www.cs.umass.edu/lfw/`)
still has the full download page, and it states the same numbers used above:

- `lfw.tgz` (original, unaligned): 173MB, MD5 `a17d05bd522c52d84eca14327a23d494`.
- `lfw-funneled.tgz`: 233MB, MD5 `1b42dfed7d15c9b2dd63d5e5840c86ad`.
- `lfw-deepfunneled.tgz`: 111MB, MD5 `68331da3eb755a505a502b5aacb3c201`.
- `pairs.txt`, `people.txt`, `pairsDevTrain.txt`, `pairsDevTest.txt`, `peopleDevTrain.txt`,
  `peopleDevTest.txt` for the View 1/View 2 splits described in the archived `README.txt`
  (13,233 images, 5,749 people).

A memento of the actual `lfw-deepfunneled.tgz` binary from 2023-11-18 still serves:
`https://web.archive.org/web/20231118124720if_/https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz`
answers `200`, `content-type: application/x-gzip`, `content-length: 108761145`, `accept-ranges:
bytes`. That's close enough to the site's rounded "111MB" to be the same file, but Wayback gives
no checksum header to cross-check against the published MD5, and archive.org's own "available"
API (`http://archive.org/wayback/available?url=...`) reports this URL as having *no* archived
snapshot even though the direct memento URL clearly serves one. Don't build automation around
Wayback URLs for this; use it only as a manual fallback.

### LFW: scikit-learn / figshare mirror (recommended)

`sklearn/datasets/_lfw.py` (checked at the `main` branch,
`https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py`) defines the
real download targets scikit-learn uses, and none of them point at UMass:

```
ARCHIVE = RemoteFileMetadata(
    filename="lfw.tgz",
    url="https://ndownloader.figshare.com/files/5976018",
    checksum="055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",  # sha256
)
FUNNELED_ARCHIVE = RemoteFileMetadata(
    filename="lfw-funneled.tgz",
    url="https://ndownloader.figshare.com/files/5976015",
    checksum="b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a",  # sha256
)
```

plus `pairsDevTrain.txt` (`5976012`), `pairsDevTest.txt` (`5976009`), and `pairs.txt` (`5976006`).
These checksums are sha256; scikit-learn verifies against sha256, torchvision (below) publishes
md5 for the same files. Both are correct, just different digests of the same bytes: a HEAD/range
request against every figshare URL returned a 302 to a presigned S3 URL, and a 1-byte range GET
against that S3 URL returned an `ETag` equal to torchvision's published MD5 in every case:

| file | figshare id | bytes | ETag / MD5 |
|---|---|---|---|
| lfw.tgz | 5976018 | 180,566,744 | `a17d05bd522c52d84eca14327a23d494` |
| lfw-funneled.tgz | 5976015 | 243,346,528 | `1b42dfed7d15c9b2dd63d5e5840c86ad` |
| pairsDevTrain.txt | 5976012 | 56,579 | `4f27cbf15b2da4a85c1907eb4181ad21` |
| pairsDevTest.txt | 5976009 | 26,002 | `5132f7440eb68cf58910c8a45a2ac10b` |
| pairs.txt | 5976006 | 155,335 | `9f1ba174e4e1c508ff7cdf10ac338a7d` |

That ETag match is the strongest evidence in this whole document: it confirms the figshare
mirror is byte-identical to whatever UMass originally shipped, independent of which project's
docs you trust. Interestingly, scikit-learn's own dataset description
(`sklearn/datasets/descr/lfw.rst`) no longer even mentions `vis-www.cs.umass.edu`; it tells users
to go read the Kaggle page for details, even though the code still downloads from figshare.

`sklearn.datasets.fetch_lfw_pairs` and `fetch_lfw_people` do not fetch deep-funneled images; only
`funneled` (default) and `original` are supported by scikit-learn's downloader.

### LFW: torchvision

`torchvision.datasets.LFWPeople` / `LFWPairs` (checked at
`https://github.com/pytorch/vision/blob/main/torchvision/datasets/lfw.py`) list all three
alignments with MD5s that match the ones above (original, funneled, deepfunneled), but
`download=True` now unconditionally raises:

```
raise ValueError(
    "LFW dataset is no longer available for automatic download."
    "Please download the dataset manually and place it in the specified directory."
    "A commonly used mirror is available at: "
    "https://www.kaggle.com/datasets/jessicali9530/lfw-dataset"
)
```

So torchvision is not a scriptable source for LFW at all right now, only a source of md5 values
and a pointer at the same Kaggle mirror.

### LFW: Kaggle (`jessicali9530/lfw-dataset`)

Public metadata, no auth needed to read: `https://www.kaggle.com/api/v1/datasets/view/jessicali9530/lfw-dataset`
returns `totalBytesNullable: 117149434`, `licenseNameNullable: "Other (specified in description)"`.
The description confirms this mirror ships the deep-funneled images
(`lfw-deepfunneled.zip`) plus 10 metadata files, including `pairs.csv` and `people.csv`
reformatted from the original `.txt` files. Actually downloading files (not just metadata) needs
a Kaggle account and API token (`kaggle.json` / `KAGGLE_USERNAME` + `KAGGLE_KEY`), which is the
gating point: the catalogue is public, the bytes are not.

### LFW: Hugging Face mirrors (checked, not recommended)

Two mirrors turned up in search and neither is fit for the verification protocol:

- `vilsonrodrigues/lfw` (`https://huggingface.co/datasets/vilsonrodrigues/lfw`) repackages LFW
  into `lfw_multifaces-ingestion.zip` and `lfw_multifaces-retrieval.zip`, no `pairs.txt`, and
  is tagged `license:apache-2.0`. That tag almost certainly covers the uploader's packaging code,
  not the underlying photographs, since LFW itself carries no such licence upstream. Don't cite
  it as evidence the images are Apache-licensed.
- `bitmind/lfw` (`https://huggingface.co/datasets/bitmind/lfw`) has exactly 13,233 rows
  (`image`, `filename` only), matching the original image count, but no verification pairs and
  no licence tag at all.

### CelebA: official source (CUHK MMLab)

`https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html` lists 10,177 identities, 202,599 images, 5
landmarks and 40 binary attributes per image, and links out to a Google Drive folder split into
Img / Anno / Eval sections (aligned images, identity/attribute/bbox/landmark annotations, and
train/val/test partitions). The page's licence text, which I also found verbatim in the
Hugging Face mirror's `LICENSE` file:

> The CelebA dataset is available for non-commercial research purposes only. All images of the
> CelebA dataset are obtained from the Internet which are not property of MMLAB, The Chinese
> University of Hong Kong. [...] You agree not to reproduce, duplicate, copy, sell, trade, resell
> or exploit for any commercial purposes, any portion of the images and any portion of derived
> data. You agree not to further copy, publish or distribute any portion of the CelebA dataset,
> except for internal use at a single site within the same organization. [...] The face
> identities are released upon request for research purposes only.

That last line is odd given the page's own Identities Annotations link sits in the same public
Drive folder as everything else; nobody appears to actually gate the identity file behind a
request in practice. Google Drive itself is the practical gate: per-file daily download quotas
kick in under any kind of scripted or repeated pull, independent of authentication.

### CelebA: torchvision

`torchvision.datasets.CelebA` (`https://github.com/pytorch/vision/blob/main/torchvision/datasets/celeba.py`)
still attempts a real download via `download_file_from_google_drive`, using the original UMass
Google Drive file IDs, and still requires `gdown` to be installed. Unlike LFW, this path is not
hard-disabled, but it is unreliable in practice. Two open/closed torchvision issues from 2024
document it failing outright:

- [pytorch/vision#8268](https://github.com/pytorch/vision/issues/8268): `download=True` returns a
  Google Drive "virus scan warning" HTML page instead of the zip.
  A maintainer closed it as a duplicate of #8204 and pointed at `gdown` plus a Hugging Face
  mirror as the workaround; a commenter on that thread linked `eurecom-ds/celeba` directly.
- [pytorch/vision#1920](https://github.com/pytorch/vision/issues/1920): earlier report of the same
  class of failure, described elsewhere as Google Drive's daily download quota being exceeded.

### CelebA: Hugging Face `flwrlabs/celeba` (recommended)

`https://huggingface.co/api/datasets/flwrlabs/celeba` confirms `"gated": false`. Config
`img_align+identity+attr` has an `image` column, `celeb_id`, and 40 boolean attribute columns
(`5_o_Clock_Shadow`, `Arched_Eyebrows`, ... through the standard CelebA attribute list). Splits:

| split | rows | shards |
|---|---|---|
| train | 162,770 | 19 parquet files |
| valid | 19,867 | 3 parquet files |
| test | 19,962 | 3 parquet files |

162,770 + 19,867 + 19,962 = 202,599, exactly the official CelebA image count, which is good
evidence this mirror wasn't resampled or deduplicated. A HEAD on one shard
(`https://huggingface.co/datasets/flwrlabs/celeba/resolve/main/img_align+identity+attr/train-00000-of-00019.parquet`)
302s to a signed `us.aws.cdn.hf.co` URL with no login prompt, confirming it's fetchable
unauthenticated. Total across all 25 shards, from the Hub API's `siblings[].size`, is
11,734,694,689 bytes (~11.7 GB).

### CelebA: Hugging Face `eurecom-ds/celeba` (checked, not recommended as primary)

Also ungated. Fuller schema than `flwrlabs/celeba`: `image`, `attributes` (40 x int8), `identity`,
`bbox` (4 x int32), `landmarks` (10 x int32). Same row counts (162,770 / 19,867 / 19,962).
Reported `dataset_size` is 8,929,121,333 bytes (~8.9 GB), smaller than `flwrlabs/celeba` for the
same image content, likely a difference in stored image encoding rather than a difference in
what's included.

Two reasons it isn't the primary pick: its README carries no licence tag or licence text at all
(compare `flwrlabs/celeba`'s explicit `LICENSE` file), and its own porting script builds `bbox`
straight from `torchvision.datasets.CelebA(target_type=["attr","identity","bbox","landmarks"])`
on the *aligned* images, while torchvision's own docstring warns those bbox coordinates are for
the original uncropped images and "will not match and may fall outside the image boundaries" once
applied to the aligned crop. Ryuk doesn't need bounding boxes since CelebA's aligned images are
already cropped, so this mismatch doesn't block using the dataset, but it's a reason to prefer
the simpler mirror that doesn't carry a broken field.

### CelebA: Kaggle (`jessicali9530/celeba-dataset`)

Public metadata again reachable without auth:
`https://www.kaggle.com/api/v1/datasets/view/jessicali9530/celeba-dataset` returns
`totalBytesNullable: 1452931921` (~1.45 GB), `licenseNameNullable: "Other (specified in
description)"`. The description lists `img_align_celeba` (the image folder), `list_eval_
partition.csv`, `list_bbox_celeba.csv`, `list_landmarks_align_celeba.csv`, and
`list_attr_celeba.csv`. It does not mention an identity file. A separate, smaller, unofficial
Kaggle dataset (`kymo9890/identity-celeba`) exists purely to fill that gap, which is itself a sign
the main mirror is missing something the ticket explicitly asked for. Downloading either requires
a Kaggle account and API token even though the catalogue metadata above needed none.

## Gotchas

The official LFW host (`vis-www.cs.umass.edu`) is dead at DNS right now, not just slow or
rate-limited. Confirmed with `dig @8.8.8.8`, not just this sandbox's resolver, so it's not a local
network quirk. Any script, doc, or paper that says "download from vis-www.cs.umass.edu" needs to
be read as historical, not current.

CelebA's official distribution is Google Drive, and Google Drive enforces a shared per-file daily
download quota, not a per-account one. A CI job or repeated laptop run can trip "quota exceeded"
even with valid credentials, because someone else's downloads count against the same limit. This
is why torchvision's own maintainers now point people at Hugging Face mirrors instead of fixing
the Drive path.

Kaggle datasets are publicly readable as metadata (`kaggle.com/api/v1/datasets/view/...` needs no
auth) but not publicly downloadable: actually pulling files needs a Kaggle account and an API
token (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`). Don't assume a script that can
read a dataset's metadata can also fetch its files.

Two different LFW checksum sets are both "correct": scikit-learn's figshare mirror publishes
sha256, torchvision publishes md5, for what's confirmed above to be the same underlying files.
Verify against whichever digest your fetcher actually supports; don't treat a mismatch in digest
*type* as a mismatch in *content*.

Licence tags on Hugging Face dataset cards aren't always about the data. `vilsonrodrigues/lfw` is
tagged `license:apache-2.0`; that's very unlikely to cover face photographs LFW itself never
licensed permissively; it most likely describes the uploader's own repackaging code. Read the
actual licence file or text, not just the tag, before citing one of these mirrors in anything that
matters.

Wayback Machine mementos can serve a real binary (confirmed for `lfw-deepfunneled.tgz`) while
archive.org's own "is this archived" API says there's nothing there. Treat Wayback as a
manual, one-off fallback, not something to hardcode into a pipeline.
