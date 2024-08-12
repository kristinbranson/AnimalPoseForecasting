from flyllm.run_flyllm import main


def test_main():
    """This test just ensures that main runs without crashing. Takes about 1 minute to run.
    """
    config_file = "/groups/branson/bransonlab/test_data_apf/test_config.json"
    main(config_file)


if __name__ == "__main__":
    test_main()
