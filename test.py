def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def test_add():
    """Test the add function."""
    assert add(1, 2) == 3


if __name__ == "__main__":
    test_add()
    print("All tests passed.")
