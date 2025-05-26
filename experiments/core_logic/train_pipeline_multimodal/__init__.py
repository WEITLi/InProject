#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练流水线模块
"""

from .multimodal_trainer import MultiModalTrainer
from .multimodal_model import MultiModalAnomalyDetector

__all__ = ['MultiModalTrainer', 'MultiModalAnomalyDetector'] 