def staircase_lr(lr_values, epoch):
    """Computes the learning rate as a function of the epoch number.

    Args:
        lr_values (list(float, float)): A list of pairs (A, B), where A
            is the upper bound for the epoch number and B is the
            corresponding learning rate value. `Note`: The pairs inside
            `lr_values` must be sorted in ascending order according to
            their first element.
        epoch (int): An integer representing the epoch number.

    Raises:
        ValueError: A Value error will be raised whenever the upper
            bound values inside `lr_values` do not cover all the
            possible cases, i.e. when epoch is either larger or smaller
            than any other upper bound.

    Returns:
        float: The learning rate value as a function of the given epoch
            number.
    """
    for upper_bound, lr in lr_values:
        if epoch < upper_bound:
            return lr
    raise ValueError(
        "Upper bound values inside parameter 'lr_values' are not exhaustive."
    )