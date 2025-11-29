"""
yfinance를 사용하여 주가 데이터를 가져오는 모듈
"""

import yfinance as yf
from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import Session
from sqlalchemy import and_

from database import StockPrice


class StockDataFetcher:
    """주가 데이터 가져오기 및 DB 저장"""

    @staticmethod
    def fetch_and_save_stock_data(
        db: Session,
        symbol: str = "005930.KS",  # 삼성전자 (한국 주식)
        period: str = "1mo",        # 최근 1개월
    ) -> dict:
        """
        yfinance로 주가 데이터를 가져와서 DB에 저장

        Args:
            db: 데이터베이스 세션
            symbol: 주식 심볼 (예: "005930.KS" for 삼성전자)
            period: 기간 ("1mo", "3mo", "6mo", "1y" 등)

        Returns:
            저장된 데이터 정보
        """
        try:
            # yfinance로 데이터 가져오기
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                return {
                    "success": False,
                    "message": f"No data found for symbol: {symbol}",
                    "records_saved": 0,
                }

            # 심볼에서 .KS / .KQ 제거 (DB 저장용)
            clean_symbol = symbol.replace(".KS", "").replace(".KQ", "")

            records_saved = 0
            records_updated = 0

            # 데이터프레임을 순회하며 DB에 저장
            for date, row in df.iterrows():
                # yfinance index는 Timestamp → date()로 날짜만 사용
                date_only = date.date()

                # 기존 레코드 확인
                existing = (
                    db.query(StockPrice)
                    .filter(
                        and_(
                            StockPrice.symbol == clean_symbol,
                            StockPrice.date == date_only,
                        )
                    )
                    .first()
                )

                if existing:
                    # 업데이트
                    existing.open = float(row["Open"])
                    existing.high = float(row["High"])
                    existing.low = float(row["Low"])
                    existing.close = float(row["Close"])
                    existing.volume = float(row["Volume"])
                    # adj_close 컬럼을 StockPrice에 추가했다면 여기서도 같이 업데이트 가능
                    # existing.adj_close = float(row["Close"])
                    records_updated += 1
                else:
                    # 새로 추가
                    stock_price = StockPrice(
                        symbol=clean_symbol,
                        date=date_only,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                        # adj_close=float(row["Close"]),  # 필요하면 StockPrice 모델에 컬럼 추가
                    )
                    db.add(stock_price)
                    records_saved += 1

            db.commit()

            return {
                "success": True,
                "message": f"Successfully fetched data for {symbol}",
                "symbol": clean_symbol,
                "period": period,
                "records_saved": records_saved,
                "records_updated": records_updated,
                "total_records": records_saved + records_updated,
                "date_range": {
                    "start": df.index[0].strftime("%Y-%m-%d"),
                    "end": df.index[-1].strftime("%Y-%m-%d"),
                },
            }

        except Exception as e:
            db.rollback()
            return {
                "success": False,
                "message": f"Error fetching data: {str(e)}",
                "records_saved": 0,
            }

    @staticmethod
    def get_stock_prices(
        db: Session,
        symbol: str = "005930",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[StockPrice]:
        """
        DB에서 주가 데이터 조회

        Args:
            db: 데이터베이스 세션
            symbol: 주식 심볼
            start_date: 시작 날짜
            end_date: 종료 날짜
            limit: 최대 레코드 수

        Returns:
            주가 데이터 리스트
        """
        query = db.query(StockPrice).filter(StockPrice.symbol == symbol)

        if start_date:
            query = query.filter(StockPrice.date >= start_date.date())
        if end_date:
            query = query.filter(StockPrice.date <= end_date.date())

        query = query.order_by(StockPrice.date.desc()).limit(limit)

        return query.all()

    @staticmethod
    def get_latest_price(db: Session, symbol: str = "005930") -> Optional[StockPrice]:
        """
        가장 최근 주가 데이터 조회

        Args:
            db: 데이터베이스 세션
            symbol: 주식 심볼

        Returns:
            최근 주가 데이터
        """
        return (
            db.query(StockPrice)
            .filter(StockPrice.symbol == symbol)
            .order_by(StockPrice.date.desc())
            .first()
        )


# 전역 인스턴스
stock_fetcher = StockDataFetcher()