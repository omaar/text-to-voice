import spacy
from typing import List, Dict, Optional, Any
from text_to_voice.core.logging_config import setup_logger
logger = setup_logger("spacy_manager")

class SpacyManager:
    def __init__(self, custom_models: Dict[str, Any] = None):
        try:
            # Cargamos el modelo en español
            self.nlp = spacy.load("es_core_news_md")
            # Diccionario para almacenar modelos personalizados
            self.custom_models = custom_models or {}
            logger.info("Modelo de spaCy cargado exitosamente")
        except OSError:
            logger.warning("Modelo de spaCy no encontrado. Intente instalarlo con: python -m spacy download es_core_news_md")
            self.nlp = None

    def add_custom_model(self, model_name: str, model: Any) -> None:
        """
        Añade un modelo personalizado al pipeline de corrección
        """
        self.custom_models[model_name] = model
        logger.info(f"Modelo personalizado '{model_name}' añadido exitosamente")

    def detect_spelling_errors(self, text: str) -> List[Dict[str, Any]]:
        """
        Detecta errores ortográficos en el texto y retorna una lista de errores encontrados
        """
        if not text or not self.nlp:
            return []

        errors = []
        doc = self.nlp(text)

        # Aplicamos reglas básicas de detección de errores
        for token in doc:
            # Verificamos si la palabra está en el vocabulario del modelo
            if not token.is_punct and not token.is_space and not token.like_num:
                if not token.is_oov:  # is_oov = is out of vocabulary
                    continue
                
                error = {
                    "token": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "suggestions": self._get_suggestions(token.text)
                }
                errors.append(error)

        # Aplicamos modelos personalizados si existen
        for model_name, model in self.custom_models.items():
            try:
                custom_errors = model.detect_errors(text)
                if custom_errors:
                    errors.extend(custom_errors)
            except Exception as e:
                logger.error(f"Error al aplicar modelo personalizado {model_name}: {str(e)}", exc_info=True)

        return errors

    def _get_suggestions(self, word: str) -> List[str]:
        """
        Genera sugerencias para una palabra incorrecta
        """
        suggestions = []
        # Aquí podrías implementar algoritmos como Levenshtein distance
        # o usar servicios externos para obtener sugerencias
        return suggestions

    def correct_spelling(self, text: str) -> str:
        """
        Corrige la ortografía del texto utilizando spaCy y modelos personalizados
        """
        if not text:
            return text
            
        try:
            # Detectamos errores
            errors = self.detect_spelling_errors(text)
            logger.info(f"Errores encontrados: {errors}")
            corrected_text = text

            # Aplicamos correcciones
            for error in sorted(errors, key=lambda x: x["start"], reverse=True):
                if error["suggestions"]:
                    # Tomamos la primera sugerencia como la corrección
                    corrected_text = (
                        corrected_text[:error["start"]] +
                        error["suggestions"][0] +
                        corrected_text[error["end"]:]
                    )

            # Aplicamos modelos personalizados
            # for model in self.custom_models.values():
            #     try:
            #         corrected_text = model.correct_text(corrected_text)
            #     except Exception as e:
            #         logger.error(f"Error al aplicar corrección personalizada: {str(e)}", exc_info=True)

            return corrected_text

        except Exception as e:
            logger.error(f"Error al corregir ortografía: {str(e)}", exc_info=True)
            return text