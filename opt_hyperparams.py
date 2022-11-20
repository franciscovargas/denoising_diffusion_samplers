"""Optimal runs for each of our methods.
"""

opt_funnel = {
    'funnel': {
        '32': {
            'oudstl': {
                'sigma': 1.075,
                'alpha': 1.075,
                'm': None
            },
            'pisstl': {
                'sigma': 1.0675000000000001,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.85,
                'alpha': 1.67,
                'm': 0.9
            }
        },
        '64': {
            'oudstl': {
                'sigma': 1.075,
                'alpha': 0.6875,
                'm': None
            },
            'pisstl': {
                'sigma': 0.4158333333333334,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.85,
                'alpha': 3.7,
                'm': 0.9
            }
        },
        '128': {
            'oudstl': {
                'sigma': 1.85,
                'alpha': 0.3,
                'm': None
            },
            'pisstl': {
                'sigma': 0.7416666666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.075,
                'alpha': 2.5,
                'm': 0.9
            }
        },
        '256': {
            'oudstl': {
                'sigma': 1.4625000000000001,
                'alpha': 0.3,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.6875,
                'alpha': 3.7,
                'm': 0.9
            }
        }
    }
}

opt_lgcp = {
    'lgcp': {
        '32': {
            'oudstl': {
                'sigma': 2.1,
                'alpha': 1.5000000000000002,
                'm': None
            },
            'pisstl': {
                'sigma': 1.0675000000000001,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.1,
                'alpha': 2.5,
                'm': 0.4
            }
        },
        '64': {
            'oudstl': {
                'sigma': 2.1,
                'alpha': 0.75,
                'm': None
            },
            'pisstl': {
                'sigma': 0.7416666666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.4,
                'alpha': 2.5,
                'm': 0.4
            }
        },
        '128': {
            'oudstl': {
                'sigma': 2.1,
                'alpha': 0.9000000000000001,
                'm': None
            },
            'pisstl': {
                'sigma': 0.57875,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.4,
                'alpha': 4.5,
                'm': 0.4
            }
        },
        '256': {
            'oudstl': {
                'sigma': 2.1,
                'alpha': 1.5000000000000002,
                'm': None
            },
            'pisstl': {
                'sigma': 0.4158333333333334,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.7,
                'alpha': 4.5,
                'm': 0.4
            }
        }
    }
}

opt_ion = {
    'ion': {
        '32': {
            'oudstl': {
                'sigma': 0.6875,
                'alpha': 1.4625000000000001,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.6,
                'alpha': 3.85,
                'm': 0.6000000000000001
            }
        },
        '64': {
            'oudstl': {
                'sigma': 0.3,
                'alpha': 1.075,
                'm': None
            },
            'pisstl': {
                'sigma': 0.09,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.6,
                'alpha': 3.85,
                'm': 0.6000000000000001
            }
        },
        '128': {
            'oudstl': {
                'sigma': 0.3,
                'alpha': 0.6875,
                'm': None
            },
            'pisstl': {
                'sigma': 0.4158333333333334,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.6,
                'alpha': 3.85,
                'm': 1.0
            }
        },
        '256': {
            'oudstl': {
                'sigma': 0.6875,
                'alpha': 0.6875,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.6,
                'alpha': 3.85,
                'm': 1.0
            }
        }
    }
}

opt_lr_sonar = {
    'lr_sonar': {
        '32': {
            'oudstl': {
                'sigma': 0.3,
                'alpha': 1.6500000000000001,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.15,
                'alpha': 1.7,
                'm': 2.2
            }
        },
        '64': {
            'oudstl': {
                'sigma': 0.3,
                'alpha': 1.2,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.55,
                'alpha': 1.7,
                'm': 3.1
            }
        },
        '128': {
            'oudstl': {
                'sigma': 0.3,
                'alpha': 0.75,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.55,
                'alpha': 2.9,
                'm': 2.2
            }
        },
        '256': {
            'oudstl': {
                'sigma': 0.3,
                'alpha': 0.75,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.55,
                'alpha': 2.9,
                'm': 3.1
            }
        }
    }
}

opt_vae = {
    'vae': {
        '32': {
            'oudstl': {
                'sigma': 0.61,
                'alpha': 2.2,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': None
        },
        '64': {
            'oudstl': {
                'sigma': 0.61,
                'alpha': 1.6700000000000002,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2529166666666667,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': None
        },
        '128': {
            'oudstl': {
                'sigma': 0.61,
                'alpha': 1.1400000000000001,
                'm': None
            },
            'pisstl': {
                'sigma': 0.506,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': None
        },
        '256': {
            'oudstl': {
                'sigma': 0.61,
                'alpha': 1.6700000000000002,
                'm': None
            },
            'pisstl': {
                'sigma': 0.4158333333333334,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': None
        }
    }
}

opt_brownian = {
    'brownian': {
        '32': {
            'oudstl': {
                'sigma': 0.1,
                'alpha': 2.35,
                'm': None
            },
            'pisstl': {
                'sigma': 0.08391666666666668,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.115,
                'alpha': 4.8,
                'm': 2.2
            }
        },
        '64': {
            'oudstl': {
                'sigma': 0.1,
                'alpha': 1.8,
                'm': None
            },
            'pisstl': {
                'sigma': 0.0925,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.115,
                'alpha': 4.8,
                'm': 2.2
            }
        },
        '128': {
            'oudstl': {
                'sigma': 0.1,
                'alpha': 2.35,
                'm': None
            },
            'pisstl': {
                'sigma': 0.04245833333333334,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.115,
                'alpha': 3.75,
                'm': 2.2
            }
        },
        '256': {
            'oudstl': {
                'sigma': 0.1,
                'alpha': 1.8,
                'm': None
            },
            'pisstl': {
                'sigma': 0.0408,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 0.115,
                'alpha': 4.8,
                'm': 2.2
            }
        }
    }
}

opt_nice = {
    'nice': {
        '32': {
            'oudstl': {
                'sigma': 1.5,
                'alpha': 2.125,
                'm': None
            },
            'pisstl': {
                'sigma': 0.75,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.2,
                'alpha': 1.0,
                'm': 0.9
            }
        },
        '64': {
            'oudstl': {
                'sigma': 1.5,
                'alpha': 1.75,
                'm': None
            },
            'pisstl': {
                'sigma': 0.5875,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.2,
                'alpha': 1.0,
                'm': 1.65
            }
        },
        '128': {
            'oudstl': {
                'sigma': 1.5,
                'alpha': 1.75,
                'm': None
            },
            'pisstl': {
                'sigma': 0.42500000000000004,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.2,
                'alpha': 2.5,
                'm': 0.9
            }
        },
        '256': {
            'oudstl': {
                'sigma': 1.5,
                'alpha': 2.5,
                'm': None
            },
            'pisstl': {
                'sigma': 0.2625,
                'alpha': 1.0,
                'm': None
            },
            'oududmp': {
                'sigma': 1.2,
                'alpha': 2.5,
                'm': 1.65
            }
        }
    }
}


# concatenating dicts
dlist = [opt_nice, opt_brownian, opt_vae, opt_lr_sonar, opt_ion, opt_lgcp, opt_funnel]

dnew = dict() 

for x in dlist:
  dnew.update(x)
