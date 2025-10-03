from flyllm.run_flyllm import main


def test_main():
    """This test just ensures that main runs without crashing. Takes about 1 minute to run.
    """
    config_file = "tests/flyllm/test_config.json"
    main(config_file)


if __name__ == "__main__":
    test_main()
