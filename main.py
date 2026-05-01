from src.data.statcast_fetch import refresh_statcast_data


def main():
    refresh_statcast_data(override_refresh=True)


if __name__ == '__main__':
    main()
