"""
reset_chromadb.py - Quick fix for ChromaDB initialization errors
Run this script to reset the database if you get tenant errors
"""

import shutil
from pathlib import Path


def reset_chromadb():
    """Remove and recreate the ChromaDB database."""

    print("\n" + "=" * 60)
    print("🔧 ChromaDB Reset Utility")
    print("=" * 60)

    chroma_path = Path("./rag_storage/chroma")

    if chroma_path.exists():
        print(f"\n📁 Found database at: {chroma_path}")
        print("⚠️  WARNING: This will DELETE all indexed documents!")
        print("⚠️  You will need to re-upload your files.")

        response = input("\n❓ Continue with reset? (yes/no): ").strip().lower()

        if response not in ['yes', 'y']:
            print("\n❌ Operation cancelled")
            return False

        try:
            print("\n🗑️  Removing old database...")
            shutil.rmtree(chroma_path)
            print("✅ Old database removed")
        except Exception as e:
            print(f"\n❌ Error removing database: {e}")
            print(f"💡 Try manually deleting folder: {chroma_path.absolute()}")
            return False
    else:
        print("\nℹ️  No existing database found")

    # Create fresh directory
    print("\n📁 Creating new database directory...")
    try:
        chroma_path.mkdir(parents=True, exist_ok=True)
        print("✅ New database directory created")
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return False

    # Also ensure other directories exist
    try:
        (Path("./rag_storage/logs")).mkdir(parents=True, exist_ok=True)
        (Path("./rag_storage/cache")).mkdir(parents=True, exist_ok=True)
        print("✅ All support directories created")
    except Exception as e:
        print(f"⚠️  Warning: {e}")

    print("\n" + "=" * 60)
    print("✅ Database reset complete!")
    print("✅ You can now run: streamlit run app.py")
    print("=" * 60 + "\n")

    return True


if __name__ == "__main__":
    success = reset_chromadb()

    if success:
        print("💡 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Upload your documents again")
        print("   3. Enjoy your voice-enabled BI AI!\n")
    else:
        print("\n💡 If the error persists:")
        print("   1. Update ChromaDB: pip install --upgrade chromadb")
        print("   2. Check Python version (3.8-3.11 recommended)")
        print("   3. Try reinstalling: pip uninstall chromadb && pip install chromadb\n")