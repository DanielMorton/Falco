import argparse
from meta import read_meta
from tfrecord.write_records import write_records


def main():
    """Creates tfrecords from NA Birds training and test data."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=str,
                    help="Directory containing Cornell NABird Data")
    ap.add_argument("--size", required=False, type=int,
                    default=2000,
                    help="number of runs")
    ap.add_argument("-l", "--log", action="store_true", default=False,
                    help="Log progress to command line.")
    args = vars(ap.parse_args())
    train_meta, test_meta = read_meta(bird_dir=args['dir'])
    write_records(train_data=train_meta,
                  test_data=test_meta,
                  bird_dir=args['dir'],
                  file_size=args['size'],
                  log_progress=args['log'])


if __name__ == '__main__':
    main()
