from train import EarlyStopping


def test_early_stopping_first_call_sets_best_loss_and_does_not_stop():
    es = EarlyStopping(patience=3, delta=0.01, verbose=False)

    stopped = es.check_early_stop(val_loss=1.0)

    assert es.best_loss == 1.0
    assert es.no_improvement_count == 0
    assert es.stop_training is False
    assert stopped is None


def test_early_stopping_no_improvement_increments_counter():
    es = EarlyStopping(patience=3, delta=0.01, verbose=False)

    es.check_early_stop(val_loss=1.0)
    stopped = es.check_early_stop(val_loss=1.0)

    assert es.best_loss == 1.0
    assert es.no_improvement_count == 1
    assert es.stop_training is False
    assert stopped is False


def test_early_stopping_improvement_resets_counter():
    es = EarlyStopping(patience=3, delta=0.01, verbose=False)

    es.check_early_stop(val_loss=1.0)
    es.check_early_stop(val_loss=1.0)

    stopped = es.check_early_stop(val_loss=0.98)

    assert es.best_loss == 0.98
    assert es.no_improvement_count == 0
    assert es.stop_training is False
    assert stopped is None


def test_early_stopping_triggers_after_patience_exceeded():
    es = EarlyStopping(patience=2, delta=0.01, verbose=False)

    es.check_early_stop(val_loss=1.0)

    es.check_early_stop(val_loss=1.0)
    assert es.stop_training is False

    es.check_early_stop(val_loss=1.0)
    assert es.stop_training is False

    stopped = es.check_early_stop(val_loss=1.0)

    assert es.no_improvement_count == 3
    assert es.stop_training is True
    assert stopped is True
