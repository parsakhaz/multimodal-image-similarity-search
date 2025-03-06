import Link from 'next/link';
import { FiSearch, FiUpload, FiImage, FiFilter } from 'react-icons/fi';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-8">
      <div className="max-w-5xl w-full space-y-12">
        <header className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">
            Image Similarity Search
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            A powerful tool for searching, organizing, and exploring your image collection
            using text, visual similarity, and multimodal queries.
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FeatureCard
            icon={<FiSearch className="h-8 w-8" />}
            title="Search Images"
            description="Find images using text queries, visual similarity, or a combination of both."
            href="/search"
            cta="Start searching"
          />

          <FeatureCard
            icon={<FiUpload className="h-8 w-8" />}
            title="Upload Images"
            description="Add new images to your collection with optional background removal and metadata."
            href="/manage"
            cta="Manage uploads"
          />

          <FeatureCard
            icon={<FiImage className="h-8 w-8" />}
            title="Browse Collection"
            description="View and edit all images in your collection with detailed metadata."
            href="/images"
            cta="View images"
          />

          <FeatureCard
            icon={<FiFilter className="h-8 w-8" />}
            title="Manage Filters"
            description="Create and apply filters to organize your image collection."
            href="/manage"
            cta="Manage filters"
          />
        </div>

        <div className="bg-blue-50 p-6 rounded-xl">
          <h2 className="text-xl font-semibold mb-4">How It Works</h2>
          <div className="space-y-2">
            <p className="text-gray-700">
              This application uses advanced AI to understand both the visual content and textual descriptions of your images.
            </p>
            <p className="text-gray-700">
              The backend is powered by FastAPI with ChromaDB for vector storage, while the frontend is built with Next.js for a responsive user experience.
            </p>
            <p className="text-gray-700">
              Key features include:
            </p>
            <ul className="list-disc list-inside text-gray-700 ml-4">
              <li>CLIP embeddings for powerful image and text understanding</li>
              <li>Multimodal search combining text and image inputs</li>
              <li>Background removal options for cleaner image storage</li>
              <li>Custom metadata and filtering capabilities</li>
              <li>Automatic image captioning for improved searchability</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
}

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  href: string;
  cta: string;
}

function FeatureCard({ icon, title, description, href, cta }: FeatureCardProps) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow">
      <div className="p-6 space-y-4">
        <div className="text-blue-600">{icon}</div>
        <h2 className="text-xl font-semibold">{title}</h2>
        <p className="text-gray-600">{description}</p>
        <div>
          <Link href={href} className="inline-flex items-center gap-1 text-blue-600 font-medium hover:text-blue-800">
            {cta}
            <span aria-hidden="true">→</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
