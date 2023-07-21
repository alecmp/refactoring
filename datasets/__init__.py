from .amazon_reviews import AmazonReviewsDataset

DATASETS = {
    AmazonReviewsDataset.code(): AmazonReviewsDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
