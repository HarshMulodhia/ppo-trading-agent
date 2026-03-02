#!/usr/bin/env python3
"""
Download Data Script

Download historical market data from various sources.
"""

import argparse
import logging
from pathlib import Path

from src.data import DataCache, get_loader
from src.utils import save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Download market data."""
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument(
        "--source", type=str, default="yahoo", help="Data source (yahoo, nse, csv)"
    )
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2023-01-01", help="End date")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--cache", action="store_true", help="Cache data")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get loader
        loader = get_loader(args.source)
        logger.info(f"Loading {args.symbol} from {args.source}")

        # Load data
        data = loader.load(
            args.symbol,
            start_date=args.start,
            end_date=args.end,
        )

        # Save data
        output_file = output_dir / f"{args.symbol}_{args.start}_{args.end}.parquet"
        data.to_parquet(output_file)
        logger.info(f"Saved data to {output_file}")

        # Cache if requested
        if args.cache:
            cache = DataCache(cache_dir=str(output_dir / ".cache"))
            cache.save(f"{args.symbol}_{args.start}_{args.end}", data)
            logger.info("Data cached")

        # Log summary
        summary = {
            "symbol": args.symbol,
            "source": args.source,
            "start_date": args.start,
            "end_date": args.end,
            "records": len(data),
            "columns": list(data.columns),
        }

        save_json(summary, str(output_dir / f"{args.symbol}_summary.json"))
        logger.info(f"Summary: {summary}")

    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


if __name__ == "__main__":
    main()
